# Quick Start Guide - Version 2.0 🚀

## What's New in Version 2.0?

### 🧠 Smart Training
**Problem**: "I don't know what training settings to use!"
**Solution**: Click **"🧠 Auto-Configure Settings"** - it analyzes your dataset and sets everything perfectly!

### 🎯 Clean Detection
**Problem**: "I'm seeing duplicate boxes on the same object!"
**Solution**: Enable **"Remove Duplicates (NMS)"** checkbox - keeps only the best detection!

### 📚 Instant Help
**Problem**: "What does 'epochs' mean?"
**Solution**: Press **F1** anywhere - comprehensive guide explaining everything!

### 💾 Auto-Save
**Problem**: "I forgot to save and lost my work!"
**Solution**: Check **"Auto-save on image change"** - never lose work again!

---

## 5-Minute Tutorial

### Step 1: Create Your First Project (1 min)
```
1. Click "➕ New" button
2. Enter project name: "my_first_project"
3. Click "➕ Add Class" 
4. Add classes: "cat", "dog", "person"
```

### Step 2: Load and Annotate Images (2 min)
```
1. Go to "Label" tab
2. Click "📁 Load" to load an image
3. Press "B" for box mode
4. Draw a box around an object
5. Select its class
6. Repeat for all objects
7. Press Ctrl+S to save (or enable auto-save!)
```

### Step 3: Train with Intelligence (1 min)
```
1. Go to "Train" tab
2. Click "🧠 Auto-Configure Settings" ⭐
3. Review the smart recommendations
4. Click "🚀 Start Training"
5. Watch the progress!
```

### Step 4: Test Your Model (1 min)
```
1. Go to "Recognize" tab
2. Select your trained model
3. Adjust confidence slider (try 0.7)
4. Enable "Remove Duplicates (NMS)"
5. Click "📸 Capture & Recognize"
6. See clean results!
```

---

## Top 10 Tips for Version 2.0

### 1. 🧠 Use Auto-Training First
Don't guess at settings! Click "🧠 Auto-Configure Settings" and let the app analyze your dataset. It considers:
- Number of images
- Complexity (annotations per image)
- Number of classes
- And more!

### 2. 🎯 Adjust Confidence Threshold
Not seeing enough detections? **Lower the threshold** (try 0.3-0.5)
Seeing too many false positives? **Raise the threshold** (try 0.7-0.9)

### 3. ✅ Always Enable NMS
Unless you specifically need overlapping boxes, **keep NMS enabled**. It removes duplicates and gives you one clean box per object.

### 4. 💾 Enable Auto-Save
Check the **"Auto-save on image change"** box in Label tab. Navigate freely with ← → arrows and your work saves automatically!

### 5. 📏 Watch the Size Preview
When drawing boxes, you'll see **real-time dimensions** (e.g., "245×180 px"). This helps ensure consistent annotation sizes.

### 6. ⌨️ Master Keyboard Shortcuts
- **Ctrl+S**: Save
- **B**: Box mode
- **P**: Polygon mode
- **← →**: Navigate images
- **Ctrl+Z**: Undo
- **F1**: Help!

### 7. 📚 Press F1 for Help
Confused about anything? **Press F1** for a comprehensive guide explaining:
- What epochs are
- How learning rate works
- What batch size means
- And much more!

### 8. 🎨 Use Bright Visuals
Notice the **bright green outlines** when drawing? They're easier to see! The size preview also has a dark background for better contrast.

### 9. 📊 Check the Dashboard
Go to the **Dashboard tab** to see:
- How many images you've annotated
- Class distribution
- Total annotations
- Project statistics

### 10. 🔄 Iterate Intelligently
1. Start with **10-20 images** per class
2. Click **Auto-Configure** and train
3. Test on Recognition tab
4. Add more images where model struggles
5. Retrain with new data
6. Repeat!

---

## Common Questions

### Q: How many images do I need?
**A**: Start with 10-20 per class minimum. The Auto-Configure button will analyze your dataset and adjust settings accordingly!

### Q: Why do I see duplicate boxes?
**A**: Enable **"Remove Duplicates (NMS)"** checkbox in Recognition tab. This uses Non-Maximum Suppression to keep only the best detection.

### Q: What confidence threshold should I use?
**A**: Start with **0.5** (default). Adjust based on results:
- More false positives? Increase to 0.7-0.9
- Missing detections? Decrease to 0.3-0.5

### Q: What training settings should I use?
**A**: Click **"🧠 Auto-Configure Settings"**! It will:
- Analyze your dataset size
- Check annotation complexity
- Count your classes
- Set optimal parameters
- Explain why!

### Q: How do I know if training is working?
**A**: Watch the **loss value** in the metrics console:
- Should **decrease** over epochs
- If it's not decreasing, try lowering learning rate
- If it's jumping around wildly, try smaller learning rate
- Use Auto-Configure to get it right!

### Q: Can I lose my annotations?
**A**: Not if you enable **auto-save**! Check the box in Label tab and your work saves automatically when you navigate between images.

### Q: What's the difference between epochs, learning rate, and batch size?
**A**: Press **F1** for detailed explanations! Quick version:
- **Epochs**: How many times to see all data (more = better learning, but slower)
- **Learning Rate**: How big the learning steps are (higher = faster but less stable)
- **Batch Size**: How many images to process at once (higher = faster, needs more memory)

---

## Keyboard Shortcuts Cheat Sheet

```
FILE
━━━━━━━━━━━━━━━━━
Ctrl+S    Save
Ctrl+C    Copy
Ctrl+V    Paste

EDITING
━━━━━━━━━━━━━━━━━
Ctrl+Z    Undo
Ctrl+Y    Redo
Delete    Remove

NAVIGATION
━━━━━━━━━━━━━━━━━
←         Previous
→         Next

VIEW
━━━━━━━━━━━━━━━━━
Ctrl++    Zoom in
Ctrl+-    Zoom out
Ctrl+0    Reset zoom

MODES
━━━━━━━━━━━━━━━━━
B         Box mode
P         Polygon mode

HELP
━━━━━━━━━━━━━━━━━
F1        Training guide
```

---

## Pro Tips

### For Best Results
✅ **Balance your dataset** - Similar number of images per class
✅ **Use Auto-Configure** - Let the AI pick settings
✅ **Enable auto-save** - Never lose work
✅ **Enable NMS** - Clean, professional results
✅ **Press F1** when confused - Comprehensive help

### For Speed
⚡ **Use keyboard shortcuts** - Much faster than mouse
⚡ **Copy-paste annotations** - Ctrl+C, Ctrl+V between similar images
⚡ **Enable auto-save** - No manual saving needed
⚡ **Use presets** - Fast/Balanced/Accurate for quick starts

### For Quality
🎯 **Consistent annotation** - Same tightness for all boxes
🎯 **Good variety** - Different angles, lighting, backgrounds
🎯 **Check Dashboard** - Monitor class distribution
🎯 **Use threshold slider** - Fine-tune detection sensitivity
🎯 **Validate results** - Dashboard → Validate Annotations

---

## Next Steps

1. **Create your first project** and annotate 10-20 images
2. **Use Auto-Configure** to set training parameters
3. **Train your model** and watch the progress
4. **Test on Recognition tab** with threshold adjustment
5. **Iterate**: Add more data where model struggles
6. **Press F1** anytime you need help!

---

## Getting Help

- **Press F1** in the app for comprehensive training guide
- **Check Help → Keyboard Shortcuts** for complete reference
- **Read IMPROVEMENTS_v2.md** for technical details
- **Read CHANGELOG_v2.md** for what's new
- **Open GitHub issue** for bugs or feature requests

---

**Welcome to Version 2.0 - Now with Intelligence! 🧠🎉**

*Last Updated: 2025-10-16*
