CUDA_VISIBLE_DEVICES=1 python3 test_image_classifier.py \
    --checkpoint_path=/disk/private-data/yy/CardMatching/mobilenet_v2_train_logs_new_data/early_stop_3600/ \
    --test_list=/disk/private-data/yy/CardMatching/new_data/list_val_3600.txt \
    --label_txt=/disk/private-data/yy/CardMatching/new_data/label_list_3600.txt \
    --test_dir=/disk/private-data/yy/CardMatching/new_data/photos_new/ \
    --batch_size=16 \
    --num_classes=189 \
    --model_name=mobilenet_v2 > test_v2_new_data.out 2>&1 &
