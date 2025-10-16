# UI Changes - Compute Device Selection Feature

## Overview
This document describes the visual changes made to the Training tab to support compute device selection.

## Training Tab - Before and After

### Before (v2.0)
```
Training Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Preset: Balanced ▼]

Epochs:           [10      ]
Learning Rate:    [0.005   ]
Batch Size:       [2       ]
Momentum:         [0.9     ]
Weight Decay:     [0.0005  ]

☑ Data Augmentation

Device:           [Auto    ▼]

[🚀 Start Training]
```

### After (v3.0)
```
Training Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Preset: Balanced ▼]

Epochs:           [10      ]
Learning Rate:    [0.005   ]
Batch Size:       [2       ]
Momentum:         [0.9     ]
Weight Decay:     [0.0005  ]

☑ Data Augmentation

Compute Device: ℹ️
┌─────────────────────────────────────┐
│ [Auto (recommended)              ▼] │
│ Detected: CPU                       │
└─────────────────────────────────────┘

[🚀 Start Training]
```

## Detailed Changes

### 1. Device Selection Section (New Layout)

**Location:** Training tab, left panel, below "Data Augmentation"

**Components:**
1. **Label**: "Compute Device:" (bold, 12pt font)
2. **Info Icon**: "ℹ️" (blue color #3498DB, clickable)
3. **Dropdown Menu**: Wider (180px), modern styling
4. **Detected Device Label**: Small gray text showing hardware

**Visual Hierarchy:**
```
┌─────────────────────────────────────────────┐
│ Compute Device: ℹ️                         │  ← Header with info icon
│                                             │
│ ┌───────────────────────────────────────┐   │
│ │ Auto (recommended)                  ▼ │   │  ← Dropdown (180px wide)
│ └───────────────────────────────────────┘   │
│                                             │
│ Detected: CPU                               │  ← Device status (10pt, gray)
└─────────────────────────────────────────────┘
```

### 2. Dropdown Options

**Default State:**
```
┌───────────────────────────────────┐
│ ● Auto (recommended)              │  ← Selected (checkmark)
│   Force CPU                       │
└───────────────────────────────────┘
```

**With GPU Available:**
```
┌───────────────────────────────────┐
│ ● Auto (recommended)              │  ← Selected
│   Force GPU                       │  ← Additional option
│   Force CPU                       │
└───────────────────────────────────┘
```

### 3. Info Icon Interaction

**Before Click:**
```
Compute Device: ℹ️
                ^-- Blue color, hand cursor
```

**After Click:**
```
Status Bar (bottom):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🖥️ Device: CPU | Auto: Uses GPU if available, otherwise CPU | Force GPU: Always tries GPU... 
```

### 4. Detected Device Display States

**CPU Only System:**
```
Detected: CPU
^-- Gray color (#95A5A6), 10pt font
```

**GPU System:**
```
Detected: CUDA (NVIDIA GeForce GTX 1650)
^-- Gray color (#95A5A6), 10pt font
```

**GPU Requested but Unavailable:**
```
Detected: CPU | ⚠️ GPU requested but not available - falling back to CPU
^-- Orange color (#E67E22), 10pt font, warning icon
```

### 5. Status Bar Updates

**Bottom of Window:**

**Before:**
```
🖥️ Device: CPU | Ready
```

**During Training:**
```
🖥️ Device: CPU | 🚀 Training started on CPU
```

**GPU Warning:**
```
🖥️ Device: CPU | ⚠️ GPU requested but not available - falling back to CPU
```

### 6. Training Metrics Console

**New Log Entries:**
```
Starting training with 10 epochs
Learning rate: 0.005, Batch size: 2
Using device: CPU                          ← New line
Device preference: auto                    ← New line
--------------------------------------------------
```

**With GPU:**
```
Starting training with 10 epochs
Learning rate: 0.005, Batch size: 2
Using device: GPU (NVIDIA GeForce GTX 1650) ← Shows GPU name
Device preference: force_gpu                 ← Shows user preference
--------------------------------------------------
```

## Color Scheme

| Element | Normal | Warning | Error |
|---------|--------|---------|-------|
| Detected Device Label | #95A5A6 (Gray) | #E67E22 (Orange) | #E74C3C (Red) |
| Info Icon | #3498DB (Blue) | - | - |
| Status Messages | #95A5A6 (Gray) | #E67E22 (Orange) | #E74C3C (Red) |

## Spacing and Layout

**Frame Structure:**
```
device_frame (transparent background)
├── Row 0: Label "Compute Device:" + Info Icon
├── Row 1: Dropdown Menu (180px width)
└── Row 2: Detected Device Label (left-aligned)

Padding:
- Frame: 10px top/bottom, 5px left/right
- Between rows: 2-3px
- Inside frame: 5px all sides
```

## Responsive Behavior

1. **Dropdown width**: Fixed at 180px for consistency
2. **Detected label**: Wraps if device name is too long
3. **Info icon**: Always visible, doesn't wrap
4. **Frame**: Expands to fit content, max 2 columns in parent grid

## Accessibility

1. **Color contrast**: All text meets WCAG AA standards
2. **Interactive elements**: Clear hover states (hand cursor on info icon)
3. **Labels**: Descriptive text for screen readers
4. **Keyboard navigation**: Tab order preserved (Label → Dropdown → Button)

## Animation

1. **Dropdown**: Standard customtkinter fade-in animation
2. **Detected label**: Color change is instant (no animation)
3. **Status messages**: Fade in over 200ms

## Settings Persistence Indicator

**No explicit indicator** - Settings save silently
- User can verify by checking dropdown value after restart
- Settings file can be manually inspected at `config/settings.json`

## Edge Cases Handled

1. **Long GPU names**: Text wraps in detected label
2. **Multiple GPUs**: Shows name of GPU 0
3. **GPU becomes unavailable**: Warning appears immediately
4. **Settings file missing**: Creates with defaults silently
5. **Invalid preference**: Falls back to "Auto" without error dialog

## Future Enhancement Opportunities

1. Add GPU memory usage display
2. Add animated loading indicator during device check
3. Add tooltip on hover (not just click)
4. Add "Detected:" label color coding (green for GPU, yellow for CPU)
5. Add expandable section for GPU details (memory, compute capability)
