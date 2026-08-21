from ultralytics import YOLO
import torch

def main():
    # Check GPU availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load YOLOv8s model
    model = YOLO('yolov8s.pt')

    # Train with RTX 4060 optimized settings
    results = model.train(
        data='data.yaml',
        epochs=200,
        imgsz=640,
        batch=12,         # Optimized for 8GB VRAM
        name='rocket_detector',
        project='runs/train',
        workers=4,        # Good for laptop CPUs
        
        # Optimizer settings
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Augmentation - versatile for any orientation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=2,
        perspective=0.0001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.3,
        auto_augment='randaugment',
        erasing=0.4,
        
        # Loss settings
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # IoU for tight bounding boxes
        iou=0.7,
        
        # Training parameters
        patience=50,
        save=True,
        save_period=10,
        cache='disk',     # Cache to disk (not RAM) for laptop
        device=0,
        
        # Validation
        val=True,
        plots=True,
        
        # Performance - optimized for laptop
        amp=True,         # Mixed precision for better performance
        fraction=1.0,
        
        # Settings
        verbose=True,
        seed=0,
        deterministic=False,  # Faster on GPU
        single_cls=True,
        close_mosaic=10,
        
        # Multi-scale training (optional, uses more memory but better accuracy)
        # multi_scale=False,  # Set to True if you want better scale invariance
    )
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    # Validate on test set
    print("\nValidating on test set...")
    test_metrics = model.val(data='data.yaml', split='test')
    
    print(f"\nTest Results:")
    print(f"mAP50: {test_metrics.box.map50:.4f}")
    print(f"mAP50-95: {test_metrics.box.map:.4f}")
    print(f"Precision: {test_metrics.box.mp:.4f}")
    print(f"Recall: {test_metrics.box.mr:.4f}")
    
    # Export model
    print("\nExporting model to ONNX...")
    export_path = model.export(format='onnx', dynamic=True, simplify=True)
    print(f"Model exported to: {export_path}")
    
    # Save best model path
    best_model_path = 'runs/train/rocket_detector/weights/best.pt'
    print(f"\nBest model saved at: {best_model_path}")
    print("\nTo test tracking, run: python test_tracking.py <video_path>")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()