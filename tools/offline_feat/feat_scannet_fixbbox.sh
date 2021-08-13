# INPUT_DIR=/localdisk2/DATASET/scannet/tasks/scannet_frames_25k
OUTPUT_DIR=/localdisk2/DATASET/scannet/tasks/scannet_frames_25k_gtobjfeat
INPUT_DIR=/localdisk2/DATASET/scannet/tasks/scannet_frames_all
# OUTPUT_DIR=/localdisk2/DATASET/scannet/tasks/scannet_frames_all_gtobjfeat
BBOX_DIR=/localdisk2/DATASET/scannet/scans_train
python tools/extract_fixbox_frcn_feature_scannet.py \
--detection_cfg detectron_model.yaml \
--detection_model detectron_model.pth \
--gpu 0 \
--save_dir $OUTPUT_DIR \
--image_dir $INPUT_DIR \
--bbox_dir $BBOX_DIR