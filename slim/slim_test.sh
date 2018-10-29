CUDA_VISIBLE_DEVICES="0,1,2,3" python test_image_classifier.py \
    --checkpoint_path=/home/lifeng/tensorflow/models/research/slim/train_logs/ \
    --test_list=list_val.txt \
    --test_dir=/home/lifeng/card_matching/full_img/ \
    --batch_size=16 \
    --num_classes=91 \
    --model_name=inception_v3 > test.out 2>&1 &
