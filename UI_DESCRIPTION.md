# UI/UX Design Description - Image Labeling Studio Pro

## Visual Design Philosophy

The enhanced UI follows modern design principles with a focus on:
- **Clarity**: Every element has a clear purpose
- **Efficiency**: Minimize clicks, maximize productivity
- **Feedback**: Visual confirmation of every action
- **Consistency**: Unified design language throughout
- **Accessibility**: Keyboard navigation and clear typography

## Main Window Layout

### Window Specifications
- **Size**: 1400x900 pixels (expanded from 1100x700)
- **Title**: "Image Labeling Studio Pro"
- **Theme**: Dark mode with accent colors
- **Layout**: Vertical stack with header bar + tabbed content

## Header Bar (Project Bar)

### Visual Layout
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Project: [ProjectName]  [New Project] [Open Project] [Import] [Export] │
│                                                Statistics: 15/20 | 145   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Components
1. **Left Section** (40% width):
   - "Project:" label (bold)
   - Project name display
   - Four action buttons with 100-80px width
   - Buttons: New Project (blue) | Open Project (blue) | Import | Export

2. **Right Section** (20% width):
   - Statistics summary
   - Format: "Images: X/Y | Annotations: Z | Classes: N"
   - Updates in real-time
   - Small font, gray color

### Color Scheme
- Background: Dark frame
- Buttons: Blue accent (#1f538d)
- Text: White/light gray
- Stats: Subtle gray

## Tab View

### Tab Bar
Four tabs, horizontal layout:
```
[Recognize] [Label] [Train] [Dashboard]
```

- Active tab: Highlighted
- Inactive tabs: Subtle gray
- Large, readable font
- Clear separation

---

## Label Tab (Main Annotation Interface)

### Layout Structure
```
┌──────────────┬───────────────────────────────────────────────────┐
│              │                                                    │
│  Left Panel  │              Canvas Area                          │
│   (280px)    │                                                    │
│              │              [Image Display]                       │
│              │                                                    │
└──────────────┴───────────────────────────────────────────────────┘
```

### Left Panel (280px Fixed Width)

#### 1. Annotation Mode Section
```
┌─────────────────────────────┐
│ Annotation Mode             │
│ ○ Bounding Box (B)          │
│ ○ Polygon (P)               │
└─────────────────────────────┘
```
- Radio buttons for mode selection
- Keyboard shortcuts shown in gray

#### 2. Classes Section
```
┌─────────────────────────────┐
│ Classes                     │
│ [+ Add Class]               │
│ ┌───────────────────────┐   │
│ │ cat                   │   │
│ │ dog                   │   │
│ │ person                │   │
│ └───────────────────────┘   │
└─────────────────────────────┘
```
- Bold section header
- Green "+ Add Class" button
- Listbox with all classes
- Scrollable if many classes

#### 3. Image Controls Section
```
┌─────────────────────────────┐
│ Image Controls              │
│ [◀ Prev] 5/20 [Next ▶]     │
│ [📷 Capture] [📁 Load]      │
│ [🔍+] [🔍-] [Reset] 100%   │
│ Status: Ready               │
└─────────────────────────────┘
```
- Navigation buttons with arrows
- Image counter centered
- Capture/Load buttons with icons
- Zoom controls with live percentage
- Status label (color-coded)

#### 4. Annotations Section
```
┌─────────────────────────────┐
│ Annotations                 │
│ ┌───────────────────────┐   │
│ │ cat (box)             │   │
│ │ dog (polygon)         │   │
│ │ person (box)          │   │
│ └───────────────────────┘   │
│ [Copy] [Paste]              │
│ [Undo] [Redo] [Delete]      │
│ [💾 Save Annotations]       │
└─────────────────────────────┘
```
- Scrollable list of annotations
- Shows type in parentheses
- Copy/Paste buttons (side by side)
- Undo/Redo/Delete buttons
- Large green "Save" button at bottom

### Canvas Area (Expandable)

#### Visual Elements
- **Background**: Dark gray/black (#1a1a1a)
- **Image**: Centered, maintains aspect ratio
- **Cursor**: Crosshair in annotation mode
- **Annotations**: Colored boxes/polygons with labels

#### Annotation Visualization
```
┌─────────────────────────────────────────┐
│                                         │
│    ┌────────────────┐                   │
│    │ cat            │                   │
│    │                │                   │
│    │     [Image]    │                   │
│    │                │                   │
│    └────────────────┘                   │
│                                         │
│           /\  person                    │
│          /  \                           │
│         /____\                          │
│                                         │
└─────────────────────────────────────────┘
```

#### Box Annotations
- **Border**: Rounded corners (8px radius)
- **Width**: 3px normal, 5px when selected
- **Color**: From BOX_COLORS array
- **Fill**: Semi-transparent (40% opacity)
- **Label Background**: Solid color (75% opacity)
- **Label Text**: White, readable font

#### Polygon Annotations
- **Points**: Visible when drawing (white circles)
- **Lines**: White dashed while drawing, solid when complete
- **Border**: Same style as boxes
- **Fill**: Semi-transparent colored fill
- **Completion**: Right-click to finish

---

## Train Tab

### Layout Structure
```
┌──────────────────────┬──────────────────────┐
│                      │                      │
│  Training Settings   │  Training Progress   │
│                      │                      │
└──────────────────────┴──────────────────────┘
```

### Left Column: Training Settings

#### Configuration Section
```
┌─────────────────────────────┐
│ Training Configuration      │
│                             │
│ Preset: [Balanced ▼]        │
│                             │
│ Epochs:        [10]         │
│ Learning Rate: [0.005]      │
│ Batch Size:    [2]          │
│ Momentum:      [0.9]        │
│ Weight Decay:  [0.0005]     │
│                             │
│ ☑ Data Augmentation         │
│                             │
│ [🚀 Start Training]         │
└─────────────────────────────┘
```

- Large title at top
- Dropdown for presets
- Grid layout for parameters
- Labels left, inputs right
- Checkbox for augmentation
- Large green "Start Training" button (40px height)

### Right Column: Training Progress

#### Progress Section
```
┌─────────────────────────────┐
│ Training Progress           │
│                             │
│ Training Epoch 5/10         │
│ ████████░░░░░░░░░ 50%      │
│                             │
│ Training Metrics            │
│ ┌───────────────────────┐   │
│ │ Starting training...  │   │
│ │ Loaded 15 images      │   │
│ │ Device: CPU           │   │
│ │ -------------------   │   │
│ │ Epoch 1/10 - Loss:..  │   │
│ │ Epoch 2/10 - Loss:..  │   │
│ └───────────────────────┘   │
└─────────────────────────────┘
```

- Status label (color-coded)
- Progress bar (full width)
- Percentage display below bar
- Scrollable metrics console
- Dark background console (#2b2b2b)
- Monospace font (Consolas, 10pt)
- Auto-scroll to bottom

---

## Dashboard Tab

### Layout Structure
```
┌─────────────────────────────────────────────┐
│         Project Dashboard                   │
│                                             │
│ ┌────────┬────────┬────────┬────────┐      │
│ │ Total  │Annotated│  Total │Classes│      │
│ │ Images │ Images │Annot.  │       │      │
│ │   20   │   15   │  145   │   3   │      │
│ └────────┴────────┴────────┴────────┘      │
│                                             │
│ ┌──────────────────┬───────────────────┐   │
│ │ Class            │ Quick Actions     │   │
│ │ Distribution     │                   │   │
│ │                  │ [Refresh Stats]   │   │
│ │ cat: 50 ████     │ [Validate Annot.] │   │
│ │ dog: 60 █████    │ [Export Report]   │   │
│ │ person: 35 ███   │ [Backup Project]  │   │
│ │                  │                   │   │
│ │                  │ [Help/Guide]      │   │
│ └──────────────────┴───────────────────┘   │
└─────────────────────────────────────────────┘
```

### Components

#### 1. Statistics Cards (Top Row)
Four equal-width cards showing:
- **Total Images**: Large number
- **Annotated Images**: Large number
- **Total Annotations**: Large number
- **Classes**: Large number

Each card has:
- Small title label
- Large bold number (24pt)
- Subtle border

#### 2. Class Distribution (Bottom Left)
```
Class Distribution:

cat        : 50 (34.5%) ███████
dog        : 60 (41.4%) █████████
person     : 35 (24.1%) █████
```
- Monospace font for alignment
- Bar graph using █ characters
- Percentage calculations
- Sorted by count (descending)

#### 3. Quick Actions (Bottom Right)
- Stacked buttons
- Full width within panel
- Standard button styling
- Clear action labels

#### 4. Help Text (Bottom Right)
```
Welcome to Image Labeling Studio Pro!

Quick Start:
1. Create or open a project
2. Add your classes
3. Load or capture images
4. Draw annotations
5. Train your model

Keyboard Shortcuts:
Ctrl+S: Save
Ctrl+Z: Undo
...
```
- Scrollable textbox
- Friendly tone
- Step-by-step guide
- Shortcut reference

---

## Recognize Tab (Existing, Enhanced)

### Layout
```
┌──────────────┬───────────────────────────────┐
│              │                               │
│ Recognizers  │      Canvas Area              │
│              │                               │
│ [Model ▼]    │      [Recognized Image]       │
│              │                               │
│ [Capture &   │                               │
│  Recognize]  │                               │
│              │                               │
│ Status       │                               │
│              │                               │
│ Results      │                               │
│ ┌──────────┐ │                               │
│ │ cat 95%  │ │                               │
│ │ dog 87%  │ │                               │
│ └──────────┘ │                               │
│              │                               │
│ [Save Result]│                               │
│ [Copy Labels]│                               │
└──────────────┴───────────────────────────────┘
```

---

## Color Palette

### Primary Colors
- **Background Dark**: #1a1a1a
- **Panel Dark**: #2b2b2b
- **Frame Gray**: #3b3b3b
- **Text White**: #ffffff
- **Text Gray**: #cccccc

### Accent Colors
- **Blue (Primary)**: #1f538d
- **Green (Success)**: #27ae60, darkgreen
- **Orange (Warning)**: #e67e22
- **Red (Error)**: #e74c3c
- **Light Blue (Info)**: #4a90e2

### Status Colors
- **Ready/Idle**: Gray (#808080)
- **In Progress**: Orange (#e67e22)
- **Success**: Green (#27ae60)
- **Error**: Red (#e74c3c)
- **Info**: Light Blue (#4a90e2)

### Annotation Colors (BOX_COLORS array)
1. #FF5733 (Red-Orange)
2. #33FF57 (Green)
3. #3357FF (Blue)
4. #F3FF33 (Yellow)
5. #FF33E3 (Magenta)
6. #33FFF4 (Cyan)
7. #FFA533 (Orange)
8. #8D33FF (Purple)
9. #33FF8D (Light Green)
10. #FF3380 (Pink)

---

## Typography

### Font Families
- **Primary**: System default (Segoe UI on Windows)
- **Monospace**: Consolas, Courier New
- **Buttons**: CTkFont (CustomTkinter default)

### Font Sizes
- **Title**: 24pt, bold
- **Section Headers**: 17-18pt, bold
- **Subsections**: 16pt, bold
- **Buttons**: 14pt
- **Body Text**: 12pt
- **Status**: 11-12pt
- **Labels**: 11pt
- **Console**: 10pt monospace

---

## Interactive Elements

### Buttons

#### Primary Buttons (Large Actions)
- **Size**: 40px height, full width
- **Color**: Green (#27ae60)
- **Hover**: Dark green
- **Examples**: "Start Training", "Save Annotations"

#### Secondary Buttons (Standard Actions)
- **Size**: Default height (~30px)
- **Color**: Blue (default theme)
- **Hover**: Darker blue
- **Examples**: "Add Class", "Load Image"

#### Small Buttons (Utility)
- **Size**: Compact, 40-60px width
- **Color**: Default theme
- **Examples**: Zoom controls, Undo/Redo

### Input Fields
- **Height**: Standard (~30px)
- **Width**: 100px for numbers, full width for text
- **Style**: Light border, dark background
- **Placeholder**: Gray text

### Lists
- **Background**: Lighter than panel
- **Selection**: Blue highlight
- **Font**: 12pt
- **Scrollbar**: Always visible when needed

### Progress Bar
- **Height**: 20px
- **Color**: Blue gradient
- **Style**: Rounded corners
- **Animation**: Smooth fill

---

## User Interaction Feedback

### Visual Feedback
1. **Button Click**: Slight color change
2. **Hover**: Color darkening
3. **Selection**: Blue highlight
4. **Drag**: Dashed preview
5. **Status**: Color-coded messages

### Animations
- **Smooth transitions**: No jarring changes
- **Progress bar**: Animated fill
- **Status updates**: Fade color changes
- **Canvas updates**: Immediate refresh

### Cursor States
- **Default**: Arrow
- **Annotation mode**: Crosshair
- **Pan mode**: Move/hand
- **Hover button**: Pointer

---

## Responsive Behavior

### Window Resize
- Canvas adapts to available space
- Left panel stays fixed width (280px)
- Image maintains aspect ratio
- Annotations scale with image

### Content Overflow
- Scrollbars appear as needed
- Lists scroll vertically
- Console scrolls vertically
- No horizontal scrolling

---

## Accessibility Features

### Keyboard Navigation
- Tab through interactive elements
- Enter to activate buttons
- Arrow keys for lists
- Full keyboard shortcuts

### Visual Clarity
- High contrast text
- Clear button labels
- Status indicators
- Color + text for states (not color alone)

### Error Prevention
- Confirmation for destructive actions
- Clear status messages
- Validation before operations
- Undo support

---

## Summary

The enhanced UI transforms the application from a basic tool to a professional-grade platform through:
- **Larger workspace**: Better for detailed work
- **Organized layout**: Logical grouping of features
- **Visual hierarchy**: Important actions stand out
- **Rich feedback**: Always know what's happening
- **Modern aesthetics**: Professional appearance
- **Efficient workflow**: Minimal clicks, maximum productivity

The design balances **power** (many features) with **simplicity** (easy to learn and use).
