# Next Steps: Getting Real Kubric/MOVi Data

## Current Status ✅

**Training pipeline is fully functional!**
- ✅ Completed 5-step training with dummy data
- ✅ All components verified: data loading, model forward/backward, loss, validation
- ✅ Training speed: ~1.76 it/s on GPU
- ✅ Code committed and pushed

**What's working:**
```bash
# Quick test (5 steps, ~3 seconds)
python scripts/train_simple.py \
    --model-config configs/model/vit_b_tiny.yaml \
    --training-config configs/training/quick_test.yaml \
    --data-dir data/kubric
```

## Next: Get Real Data

You have **3 options** for getting real Kubric/MOVi data:

---

### Option 1: Download MOVi via TensorFlow Datasets (⭐ Recommended)

**Easiest and fastest** - pre-generated synthetic datasets from Google

#### Step 1: Install TensorFlow Datasets
```bash
pip install tensorflow-datasets tensorflow
```

#### Step 2: Download MOVi-A (simplest dataset)
```python
import tensorflow_datasets as tfds

# Download MOVi-A dataset (256x256 resolution)
ds = tfds.load('movi_a/256x256', split='train',
               data_dir='./data/movi',
               download=True)

print("Download complete!")
```

#### Step 3: Convert to D4RT format
You'll need to create a converter script (I can help with this):
```bash
python scripts/convert_movi_to_kubric.py \
    --input-dir data/movi/downloads/movi_a \
    --output-dir data/kubric
```

**Pros:**
- ✅ Pre-generated, high quality
- ✅ No rendering needed
- ✅ Multiple difficulty levels (A=easy, E=hard)

**Cons:**
- ⚠️ Requires ~50-100GB disk space
- ⚠️ Needs conversion script (we'll create this)

**Dataset info:**
- MOVi-A: Simple scenes, few objects
- MOVi-B: More complex
- MOVi-C: Even more complex
- MOVi-D: Indoor scenes
- MOVi-E: Outdoor scenes

---

### Option 2: Generate Kubric Locally

**Full control** - generate custom scenes

#### Requirements:
```bash
# 1. Install Docker
sudo apt install docker.io
sudo usermod -aG docker $USER  # Add yourself to docker group
# Log out and back in

# 2. Install Blender (3D rendering engine)
sudo snap install blender --classic

# 3. Install Kubric
pip install git+https://github.com/google-research/kubric.git
```

#### Generate scenes:
```bash
python scripts/prepare_kubric_data.py --method generate --num-scenes 100
```

**Pros:**
- ✅ Full control over scene complexity
- ✅ Can generate unlimited data
- ✅ Custom scenes for specific tasks

**Cons:**
- ⚠️ Requires Docker + Blender setup
- ⚠️ Very slow (minutes per scene)
- ⚠️ High compute requirements

**Generation speed:**
- ~2-5 minutes per scene (CPU)
- 100 scenes = ~3-8 hours

---

### Option 3: Use Other Datasets

**Alternative datasets** that work with D4RT:

#### PointOdyssey (Real videos with point tracks)
- Download: https://pointodyssey.com/
- Has ground truth point tracks
- Real-world videos
- Requires account creation (FREE)

#### ScanNet/ScanNet++ (Indoor RGB-D)
- Download: http://www.scan-net.org/
- Requires academic license (FREE)
- Real indoor scenes with depth

#### Co3Dv2 (Multi-view objects)
- Download: https://github.com/facebookresearch/co3d
- No account needed
- Good for object-centric tasks

---

## Recommended Approach

**Start with MOVi-A via TensorFlow Datasets:**

1. **Download MOVi-A** (easiest dataset)
   ```bash
   python << EOF
   import tensorflow_datasets as tfds
   ds = tfds.load('movi_a/256x256', split='train[:100]',  # Just 100 samples
                  data_dir='./data/movi', download=True)
   print("✓ Downloaded 100 training samples")
   EOF
   ```

2. **Let me know when download completes**
   - I'll create the conversion script
   - Convert MOVi format → D4RT format

3. **Train ViT-B on real data**
   ```bash
   python scripts/train_simple.py \
       --model-config configs/model/vit_b.yaml \
       --training-config configs/training/debug.yaml \
       --data-dir data/kubric
   ```

---

## If You Need to Create Accounts

**Use Claude in Chrome** for:
- PointOdyssey account creation
- ScanNet academic license application
- Any web-based downloads

Just let me know which dataset you want and I can guide you through it!

---

## Quick Commands Reference

```bash
# Check current dummy data
python scripts/prepare_kubric_data.py --method check

# Create more dummy data (for testing)
python scripts/prepare_kubric_data.py --method dummy --num-scenes 50

# Quick training test (5 steps)
python scripts/train_simple.py \
    --model-config configs/model/vit_b_tiny.yaml \
    --training-config configs/training/quick_test.yaml \
    --data-dir data/kubric

# Full training (once you have real data)
python scripts/train_simple.py \
    --model-config configs/model/vit_b.yaml \
    --training-config configs/training/debug.yaml \
    --data-dir data/kubric
```

---

## What's Next?

**Choose one:**
1. 📥 **Download MOVi-A** (I recommend this - fastest path)
2. 🔧 **Generate Kubric** (if you want custom scenes)
3. 🌐 **Download other datasets** (PointOdyssey, ScanNet, etc.)

Let me know which option you prefer and I'll guide you through it!
