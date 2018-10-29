CUDA_VISIBLE_DEVICES=2 python test_image_classifier.py \
    --checkpoint_path=/home/yy/PreResearch/card_matching-master/slim/mobilenet_v1_train_logs/early_stop_3600/ \
    --test_list=data/list_val_3600.txt \
    --label_txt=data/label_list_3600.txt \
    --test_dir=../../photos/ \
    --batch_size=16 \
    --num_classes=10 \
    --model_name=mobilenet_v1 > test.out 2>&1 &
