# D4RT Test Scripts

This directory contains test and validation scripts for the D4RT implementation.

## Available Scripts

### 1. Model Testing

#### `test_model.py` - Comprehensive Model Validation

Tests all core model components:
- Model initialization for all configs (ViT-B, ViT-L, ViT-g)
- Forward pass with dummy data
- Encoder and decoder separation
- Loss computation
- Gradient flow
- Parameter count verification

**Usage:**
```bash
python scripts/test_model.py
```

**Expected Output:**
```
D4RT MODEL VALIDATION TESTS
================================================================================
TEST 1: Model Initialization
...
✓ All tests passed!
```

### 2. Data Pipeline Testing

#### `test_data.py` - Data Pipeline Validation

Tests the complete data pipeline:
- Query sampling strategy (50/25/25 distribution)
- Data augmentation transforms
- Camera parameter utilities
- Batch collation
- Patch extraction

**Usage:**
```bash
python scripts/test_data.py
```

### 3. Model Information

#### `show_model_info.py` - Display Model Architecture

Shows detailed information about D4RT models including:
- Architecture breakdown
- Parameter counts
- Memory usage estimates
- Model comparisons

**Usage:**
```bash
# Show all models
python scripts/show_model_info.py --model all

# Show specific model
python scripts/show_model_info.py --model vit_b

# Compare models
python scripts/show_model_info.py --compare

# Show memory estimates
python scripts/show_model_info.py --model vit_g --memory

# Show everything
python scripts/show_model_info.py --model all --compare --memory
```

**Example Output:**
```
D4RT MODEL: VIT-B
================================================================================
Encoder:       86,470,656 parameters
Query Encoder: 1,841,408 parameters
Decoder:       144,443,393 parameters
------------------------------------------------------------
Total:         232,755,457 parameters
================================================================================
```

## Running Tests

### Quick Validation

Run both test suites to validate the implementation:

```bash
# Test model components
python scripts/test_model.py

# Test data pipeline
python scripts/test_data.py
```

### Before Training

Before starting training, verify:

1. **Model builds correctly:**
   ```bash
   python scripts/show_model_info.py --model vit_b
   ```

2. **All tests pass:**
   ```bash
   python scripts/test_model.py && python scripts/test_data.py
   ```

3. **Memory requirements:**
   ```bash
   python scripts/show_model_info.py --model vit_g --memory
   ```

## Test Coverage

### Model Tests (`test_model.py`)

- ✅ Model initialization
- ✅ Forward pass
- ✅ Encoder output shape
- ✅ Decoder output shape
- ✅ Loss computation
- ✅ Gradient flow
- ✅ Parameter counts

### Data Tests (`test_data.py`)

- ✅ Query sampling
- ✅ Query coordinate ranges
- ✅ Data transforms
- ✅ Camera utilities
- ✅ Batch collation
- ✅ Patch extraction

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the project root:
```bash
cd /path/to/D4RT_MasImpl
python scripts/test_model.py
```

### CUDA Out of Memory

For ViT-g testing, you may need to:
- Reduce batch size
- Enable gradient checkpointing
- Use a GPU with more memory

### Test Failures

If tests fail:
1. Check the error message for specific issues
2. Verify all dependencies are installed
3. Make sure PyTorch is properly installed with CUDA support (if using GPU)

## Expected Test Times

On typical hardware:

| Test | CPU | GPU |
|------|-----|-----|
| Model Tests | ~2-5 min | ~30-60 sec |
| Data Tests | ~10-20 sec | ~5-10 sec |

## Next Steps

After all tests pass:
1. Implement training infrastructure (Phase 5)
2. Test with small model on toy data
3. Prepare datasets for full training
4. Launch training runs
