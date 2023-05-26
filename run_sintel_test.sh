mkdir -p checkpoints
python3 -u evaluate_FlowFormer.py --model checkpoints/sintel.pth --dataset sintel
python3 -u evaluate_FlowFormer.py --model checkpoints/sintel.pth --dataset sintel_submission

# python3 -u evaluate_FlowFormer.py --model checkpoints/sintel.pth --dataset kitti
# python3 -u evaluate_FlowFormer.py --model checkpoints/sintel.pth --dataset kitti_submission
