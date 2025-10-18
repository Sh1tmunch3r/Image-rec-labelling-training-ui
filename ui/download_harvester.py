"""
Image Download Harvester
A CustomTkinter-based UI for downloading images from URLs and integrating them into projects.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import asyncio
import aiohttp
import os
import re
import json
from pathlib import Path
from typing import List, Optional, Callable
from urllib.parse import urlparse, unquote
import threading
from datetime import datetime
import hashlib


# Maximum concurrent downloads
MAX_CONCURRENT_DOWNLOADS = 5


class ImageDownloadHarvester(ctk.CTkToplevel):
    """Image downloader UI that integrates with project structure"""
    
    def __init__(self, parent=None, project_path: Optional[str] = None, 
                 on_complete_callback: Optional[Callable] = None):
        """
        Initialize the image downloader.
        
        Args:
            parent: Parent window (CTk instance)
            project_path: Path to the project folder where images will be saved
            on_complete_callback: Callback to refresh UI after downloads complete
        """
        super().__init__(parent)
        
        self.project_path = project_path
        self.on_complete_callback = on_complete_callback
        self.download_queue = []
        self.downloading = False
        self.cancel_requested = False
        self.downloaded_count = 0
        self.failed_count = 0
        
        self.title("Image Download Harvester")
        self.geometry("800x700")
        
        # Make window modal if parent exists
        if parent:
            self.transient(parent)
            self.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        
        # Main container
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Header
        header = ctk.CTkLabel(main_frame, 
                             text="🌐 Image Download Harvester",
                             font=ctk.CTkFont(size=20, weight="bold"))
        header.pack(pady=(10, 5))
        
        # Project info
        if self.project_path:
            project_name = os.path.basename(self.project_path)
            project_info = ctk.CTkLabel(main_frame,
                                       text=f"📁 Project: {project_name}",
                                       font=ctk.CTkFont(size=12),
                                       text_color="#3498DB")
            project_info.pack(pady=(0, 10))
        else:
            no_project = ctk.CTkLabel(main_frame,
                                     text="⚠️ No project selected - standalone mode",
                                     font=ctk.CTkFont(size=12),
                                     text_color="#E67E22")
            no_project.pack(pady=(0, 10))
            
            # Add project selection button
            select_btn = ctk.CTkButton(main_frame,
                                      text="Select Project Folder",
                                      command=self.select_project_folder,
                                      height=32,
                                      corner_radius=8)
            select_btn.pack(pady=5)
        
        # URL input section
        url_frame = ctk.CTkFrame(main_frame)
        url_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(url_frame,
                    text="📝 Image URLs (one per line):",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        
        # Text area for URLs
        self.url_text = ctk.CTkTextbox(url_frame,
                                       height=200,
                                       font=ctk.CTkFont(size=11),
                                       wrap="word")
        self.url_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Load from file button
        load_btn = ctk.CTkButton(url_frame,
                                text="📂 Load URLs from File",
                                command=self.load_urls_from_file,
                                height=32,
                                corner_radius=8)
        load_btn.pack(pady=5)
        
        # Options frame
        options_frame = ctk.CTkFrame(main_frame)
        options_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(options_frame,
                    text="⚙️ Options:",
                    font=ctk.CTkFont(size=13, weight="bold")).pack(pady=5)
        
        # Preserve filenames option
        self.preserve_names_var = tk.BooleanVar(value=False)
        preserve_check = ctk.CTkCheckBox(options_frame,
                                        text="Preserve original filenames (when safe)",
                                        variable=self.preserve_names_var,
                                        font=ctk.CTkFont(size=11))
        preserve_check.pack(pady=2)
        
        # Use selenium fallback option
        self.use_selenium_var = tk.BooleanVar(value=False)
        selenium_check = ctk.CTkCheckBox(options_frame,
                                        text="Use Selenium for dynamic pages (slower)",
                                        variable=self.use_selenium_var,
                                        font=ctk.CTkFont(size=11))
        selenium_check.pack(pady=2)
        
        # Progress section
        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.pack(fill="x", padx=10, pady=10)
        
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        
        self.status_label = ctk.CTkLabel(progress_frame,
                                        text="Ready to download",
                                        font=ctk.CTkFont(size=11),
                                        text_color="gray")
        self.status_label.pack(pady=5)
        
        # Log area
        log_frame = ctk.CTkFrame(main_frame)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(log_frame,
                    text="📋 Download Log:",
                    font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)
        
        self.log_text = ctk.CTkTextbox(log_frame,
                                      height=120,
                                      font=ctk.CTkFont(size=10),
                                      wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Button frame
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=10)
        
        self.download_btn = ctk.CTkButton(button_frame,
                                         text="⬇️ Start Download",
                                         command=self.start_download,
                                         height=40,
                                         corner_radius=10,
                                         fg_color="#27AE60",
                                         hover_color="#229954",
                                         font=ctk.CTkFont(size=14, weight="bold"))
        self.download_btn.pack(side="left", expand=True, fill="x", padx=5)
        
        self.cancel_btn = ctk.CTkButton(button_frame,
                                       text="❌ Cancel",
                                       command=self.cancel_download,
                                       height=40,
                                       corner_radius=10,
                                       fg_color="#E74C3C",
                                       hover_color="#C0392B",
                                       font=ctk.CTkFont(size=14, weight="bold"),
                                       state="disabled")
        self.cancel_btn.pack(side="left", expand=True, fill="x", padx=5)
        
    def select_project_folder(self):
        """Select a project folder for saving images"""
        folder = filedialog.askdirectory(title="Select Project Folder")
        if folder:
            self.project_path = folder
            self.log(f"Selected project: {os.path.basename(folder)}")
            
    def load_urls_from_file(self):
        """Load URLs from a text file"""
        filepath = filedialog.askopenfilename(
            title="Select URL File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.url_text.delete("1.0", "end")
                    self.url_text.insert("1.0", content)
                    self.log(f"Loaded URLs from {os.path.basename(filepath)}")
            except Exception as e:
                self.log(f"Error loading file: {str(e)}", error=True)
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def log(self, message: str, error: bool = False):
        """Add a message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "❌ ERROR" if error else "ℹ️"
        log_message = f"[{timestamp}] {prefix} {message}\n"
        
        self.log_text.insert("end", log_message)
        self.log_text.see("end")
        
    def get_next_image_number(self) -> int:
        """Get the next available image number in the project"""
        if not self.project_path:
            return 1
            
        images_dir = os.path.join(self.project_path, "images")
        if not os.path.exists(images_dir):
            return 1
            
        # Find existing img_XXXX.* files
        existing_numbers = []
        for filename in os.listdir(images_dir):
            match = re.match(r'img_(\d+)\.\w+', filename)
            if match:
                existing_numbers.append(int(match.group(1)))
                
        return max(existing_numbers) + 1 if existing_numbers else 1
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to be safe for the filesystem"""
        # Remove or replace unsafe characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        filename = filename.strip('. ')
        # Limit length
        name, ext = os.path.splitext(filename)
        if len(name) > 200:
            name = name[:200]
        return name + ext
    
    def get_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = os.path.basename(path)
        
        # If no filename or extension, generate one
        if not filename or '.' not in filename:
            # Use URL hash as filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"downloaded_{url_hash}.jpg"
            
        return self.sanitize_filename(filename)
    
    def generate_project_filename(self, original_filename: str, image_number: int) -> str:
        """
        Generate a filename following project naming convention.
        
        Args:
            original_filename: Original filename from URL
            image_number: Sequential number for img_XXXX format
            
        Returns:
            Final filename to use
        """
        if self.preserve_names_var.get():
            # Try to preserve original name, but check for collisions
            images_dir = os.path.join(self.project_path, "images")
            safe_name = self.sanitize_filename(original_filename)
            target_path = os.path.join(images_dir, safe_name)
            
            if not os.path.exists(target_path):
                return safe_name
        
        # Use project naming convention img_XXXX.ext
        ext = os.path.splitext(original_filename)[1] or '.jpg'
        return f"img_{image_number:04d}{ext}"
    
    def parse_urls(self) -> List[str]:
        """Parse URLs from the text area"""
        content = self.url_text.get("1.0", "end")
        urls = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line and (line.startswith('http://') or line.startswith('https://')):
                urls.append(line)
                
        return urls
    
    def start_download(self):
        """Start the download process"""
        if self.downloading:
            return
            
        urls = self.parse_urls()
        if not urls:
            messagebox.showwarning("No URLs", "Please enter at least one valid URL")
            return
            
        if not self.project_path:
            messagebox.showwarning("No Project", "Please select a project folder first")
            return
            
        # Ensure images directory exists
        images_dir = os.path.join(self.project_path, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        self.download_queue = urls
        self.downloading = True
        self.cancel_requested = False
        self.downloaded_count = 0
        self.failed_count = 0
        
        self.download_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        
        self.log(f"Starting download of {len(urls)} images...")
        self.status_label.configure(text=f"Downloading 0/{len(urls)}...")
        
        # Run download in separate thread
        thread = threading.Thread(target=self.run_download_async, daemon=True)
        thread.start()
        
    def cancel_download(self):
        """Cancel the ongoing download"""
        self.cancel_requested = True
        self.log("Cancellation requested...")
        self.status_label.configure(text="Cancelling...")
        
    def run_download_async(self):
        """Run the async download in a thread"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.download_images())
        except Exception as e:
            self.log(f"Download error: {str(e)}", error=True)
        finally:
            self.downloading = False
            self.after(0, self.download_complete)
            
    async def download_images(self):
        """Download all images with concurrency control"""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for idx, url in enumerate(self.download_queue):
                if self.cancel_requested:
                    break
                task = self.download_single_image(session, url, idx, semaphore)
                tasks.append(task)
                
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def download_single_image(self, session: aiohttp.ClientSession, 
                                   url: str, index: int, semaphore: asyncio.Semaphore):
        """Download a single image"""
        async with semaphore:
            if self.cancel_requested:
                return
                
            try:
                self.after(0, lambda: self.log(f"Downloading: {url[:60]}..."))
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # Get filename
                        original_filename = self.get_filename_from_url(url)
                        image_number = self.get_next_image_number() + index
                        filename = self.generate_project_filename(original_filename, image_number)
                        
                        # Save file
                        images_dir = os.path.join(self.project_path, "images")
                        filepath = os.path.join(images_dir, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(content)
                            
                        self.downloaded_count += 1
                        self.after(0, lambda: self.log(f"✓ Downloaded: {filename}"))
                    else:
                        raise Exception(f"HTTP {response.status}")
                        
            except Exception as e:
                self.failed_count += 1
                self.after(0, lambda: self.log(f"Failed: {url[:50]} - {str(e)}", error=True))
            
            # Update progress
            total = len(self.download_queue)
            completed = self.downloaded_count + self.failed_count
            progress = completed / total if total > 0 else 0
            self.after(0, lambda: self.progress_bar.set(progress))
            self.after(0, lambda: self.status_label.configure(
                text=f"Downloaded {completed}/{total} (Success: {self.downloaded_count}, Failed: {self.failed_count})"
            ))
    
    def download_complete(self):
        """Called when download is complete"""
        self.download_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        
        if self.cancel_requested:
            self.status_label.configure(text="Download cancelled")
            self.log("Download cancelled by user")
        else:
            self.status_label.configure(text="Download complete!")
            self.log(f"Download complete! Success: {self.downloaded_count}, Failed: {self.failed_count}")
            
        # Call completion callback if provided
        if self.on_complete_callback and self.downloaded_count > 0:
            try:
                self.on_complete_callback()
            except Exception as e:
                self.log(f"Error in completion callback: {str(e)}", error=True)
        
        if self.downloaded_count > 0:
            messagebox.showinfo(
                "Download Complete",
                f"Successfully downloaded {self.downloaded_count} images!\n"
                f"Failed: {self.failed_count}\n\n"
                f"Images saved to: {os.path.join(self.project_path, 'images')}"
            )


def main():
    """Run the downloader standalone for testing"""
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    
    root = ctk.CTk()
    root.withdraw()  # Hide root window
    
    app = ImageDownloadHarvester(root)
    app.mainloop()


if __name__ == "__main__":
    main()
