# Image Downloader UI Quick Reference

## How to Access

### Method 1: Menu Bar
```
┌────────────────────────────────────┐
│ [Tools ▼]  [Help ▼]                │
│  │                                  │
│  ├─ Image Downloader  ◄─── NEW!    │
│  └─ ...                             │
└────────────────────────────────────┘
```

### Method 2: Label Tab Button
```
┌─────────────────────────────────────────────────┐
│ Label Tab                                       │
├─────────────────────────────────────────────────┤
│ 🖼️ Navigation                                   │
│ ┌─────┬─────┬─────┐                            │
│ │  ◀  │ 0/0 │  ▶  │                            │
│ └─────┴─────┴─────┘                            │
│                                                 │
│ ┌─────────┬─────────┬───────────┐              │
│ │📷 Capture│📁 Load  │🌐 Download│ ◄─── NEW!   │
│ └─────────┴─────────┴───────────┘              │
└─────────────────────────────────────────────────┘
```

## Downloader Window Layout

```
╔═══════════════════════════════════════════════════════╗
║         🌐 Image Download Harvester                   ║
╠═══════════════════════════════════════════════════════╣
║  📁 Project: MyProject                                ║
║                                                       ║
║  📝 Image URLs (one per line):                        ║
║  ┌────────────────────────────────────────────────┐  ║
║  │ https://example.com/image1.jpg                 │  ║
║  │ https://example.com/image2.png                 │  ║
║  │ https://another-site.com/photo.jpg             │  ║
║  │                                                 │  ║
║  │                                                 │  ║
║  └────────────────────────────────────────────────┘  ║
║           [📂 Load URLs from File]                    ║
║                                                       ║
║  ⚙️ Options:                                          ║
║  ☐ Preserve original filenames (when safe)           ║
║  ☐ Use Selenium for dynamic pages (slower)           ║
║                                                       ║
║  Progress:                                            ║
║  [████████████████░░░░░░░░░░] 65%                    ║
║  Downloaded 13/20 (Success: 12, Failed: 1)           ║
║                                                       ║
║  📋 Download Log:                                     ║
║  ┌────────────────────────────────────────────────┐  ║
║  │ [14:23:45] ✓ Downloaded: img_0001.jpg         │  ║
║  │ [14:23:46] ✓ Downloaded: img_0002.png         │  ║
║  │ [14:23:47] ❌ Failed: timeout.jpg - Timeout   │  ║
║  │ [14:23:48] ✓ Downloaded: img_0003.jpg         │  ║
║  └────────────────────────────────────────────────┘  ║
║                                                       ║
║  [  ⬇️ Start Download  ]  [  ❌ Cancel  ]            ║
╚═══════════════════════════════════════════════════════╝
```

## Tooltips Added

When you hover over buttons, you'll see helpful tooltips:

- **📤 Export button**: "Export annotations to various formats (COCO, YOLO, etc.)"
- **💾 Save button (Label tab)**: "Save current image annotations to project (Ctrl+S)"
- **💾 Save button (Recognize tab)**: "Export recognized images with annotations in selected format"
- **🌐 Download button**: "Download images from URLs directly to project"

## Usage Flow

```
┌─────────────────┐
│ 1. Open Project │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ 2. Click 🌐 Download or │
│    Tools → Downloader   │
└────────┬────────────────┘
         │
         ▼
┌──────────────────────┐
│ 3. Paste URLs or     │
│    Load from File    │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 4. Configure Options │
│    (optional)        │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 5. Start Download    │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 6. Monitor Progress  │
│    View Logs         │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ 7. Images Added to   │
│    Project!          │
│    UI Auto-Refreshes │
└──────────────────────┘
```

## File Naming Convention

The downloader automatically names files to match your project:

```
Before download:
  MyProject/images/
    ├─ img_0001.png
    ├─ img_0002.png
    └─ img_0003.png

After downloading 3 new images:
  MyProject/images/
    ├─ img_0001.png
    ├─ img_0002.png
    ├─ img_0003.png
    ├─ img_0004.png  ◄── NEW
    ├─ img_0005.png  ◄── NEW
    └─ img_0006.png  ◄── NEW
```

## Error Handling

The downloader handles various error scenarios:

- ❌ **Network timeout**: Logs error, continues with next image
- ❌ **Invalid URL**: Skips and logs warning
- ❌ **Access denied**: Logs HTTP error, continues
- ❌ **No project selected**: Prompts user to select one
- ❌ **Disk full**: Stops download, shows error message

## Key Features

✅ **Fast**: Up to 5 concurrent downloads
✅ **Smart**: Automatic sequential numbering
✅ **Safe**: Filename sanitization prevents issues
✅ **Transparent**: Real-time progress and detailed logs
✅ **Integrated**: Refreshes image list automatically
✅ **Persistent**: Remembers your last project
✅ **User-friendly**: Clear tooltips and error messages

## Configuration Persistence

The app now remembers your preferences in `config/app_config.json`:

```json
{
  "last_project": "/path/to/MyProject",
  "last_model": "trained_model.pth",
  "window_geometry": "1500x950",
  "last_download_folder": "/path/to/downloads"
}
```

This makes your workflow smoother by auto-selecting your last used project!
