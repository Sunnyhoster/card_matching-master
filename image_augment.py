import Augmentor
p = Augmentor.Pipeline("train")
p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.7, min_factor=0.5, max_factor=1.1)
p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)

p.sample(10000) 