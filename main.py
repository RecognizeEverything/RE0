import os
from loguru import logger
from settings import parse_args
from pipeline.segmentation_2d import seg_with_cropformer
from pipeline.merge import merge_masks
from pipeline.clip import add_clip
from pipeline.generate_txt import generate_txt, generate_instance_txt
from pipeline.merge_clip import merge_clip
from pipeline.semantic_segmentation import semantic_segmentation
from pipeline.instance_segmentation import (
    set_pandarallel_workers,
    mask_result_mapping,
    mask_based_instance_segmantation,
)
from pipeline.visualization import visiualization


def main():
    args = parse_args()
    args.project_dir = os.path.dirname(os.path.abspath(__file__))
    print(args)
    #  分割cropformer，存储好，只需要跑一遍
    logger.info("Step1: Segmantation with Cropformer")
    seg_with_cropformer(args)
    
    # 下面两步都是帅的分割
    logger.info("Step2: Mapping Result to Point Cloud")
    all_category_result = mask_result_mapping(args)

    logger.info("Step3: Mask Based Instance Segmantation")
    set_pandarallel_workers(args.nb_workers)
    point_mask_result = mask_based_instance_segmantation(
        all_category_result, args)

    # 我的第一个合并（看投影投票）
    logger.info("Step4: Merge the masks")
    segmentation_result = merge_masks(point_mask_result, args)

    # 取clip特征，供下一步合并
    logger.info("Step5: Add dict_1")
    clip_dict = add_clip(segmentation_result, args, stage=1)

    logger.info("Step6: Clip merge")
    clip_merge_result = merge_clip(segmentation_result, clip_dict, args)

    # 合并完再取clip 用于开放词汇表
    logger.info("Step7: Add clip_2")
    merged_clip_dict = add_clip(clip_merge_result, args, stage=2)

    # 语义分割（用于测结果，没有实质意义）
    logger.info("Step8: Semantic segmentation")
    semantic_segmentation_result = semantic_segmentation(
        clip_merge_result, merged_clip_dict, args)

    # 生成txt帮助测试分割结果
    logger.info("Step9: Generate instance txt")
    generate_instance_txt(clip_merge_result, semantic_segmentation_result,
                          args)

    logger.info("Step10: Visualization")
    visiualization(args, clip_merge_result)


if __name__ == "__main__":
    main()
