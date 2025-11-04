"""
Voice-Controlled Browser Automation Script
Recognizes voice commands to control Chrome browser with various actions
"""

import speech_recognition as sr
import pyautogui
import subprocess
import time
import re
import webbrowser
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, NoSuchElementException
import threading
import queue
import os
import win32gui
import win32con
import psutil

class VoiceBrowserController:
    def __init__(self):
        """Initialize the voice browser controller"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.driver = None
        self.listening = False
        self.command_queue = queue.Queue()
       
        
        # Configure speech recognition
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.2  # Longer pause for search queries
        self.search_mode = False
        
        # Browser automation settings
        self.chrome_path = self._find_chrome_path()
        
        print("🎤 Voice Browser Controller initialized!")
        print("📋 Available commands:")
        print("   • 'Open Chrome' - Launch Chrome browser")
        print("   • 'Search [your query]' - Search on Google")
        print("   • 'Search' (then say query) - Two-step search")
        print("   • 'Click' - Click at current cursor position")
        print("   • 'Open another tab' - Open new tab")
        print("   • 'Close tab' - Close current tab")
        print("   • 'Close browser' - Close entire browser")
        print("   • 'Play video' - Play/resume video")
        print("   • 'Pause video' - Pause video")
        print("   • 'Stop listening' - Exit the program")
    
    
    def _find_chrome_path(self):
        """Find Chrome executable path"""
        possible_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Users\Dhruthi M Sathish\AppData\Local\Google\Chrome\Application\chrome.exe"
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expandvars(path)
            if os.path.exists(expanded_path):
                return expanded_path
        
        print("⚠️ Chrome not found in default locations")
        return None
    
    def find_chrome_window(self):
        """Find Chrome window handle"""
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if "Google Chrome" in window_text or "chrome" in window_text.lower():
                    windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        return windows[0] if windows else None
    
    def focus_chrome_window(self):
        """Focus Chrome window using multiple reliable methods"""
        try:
            print("🎯 Focusing Chrome window...")
            
            # Method 1: Try finding Chrome window
            chrome_hwnd = self.find_chrome_window()
            if chrome_hwnd:
                try:
                    # Force restore and focus
                    win32gui.ShowWindow(chrome_hwnd, win32con.SW_RESTORE)
                    win32gui.ShowWindow(chrome_hwnd, win32con.SW_SHOW)
                    win32gui.SetForegroundWindow(chrome_hwnd)
                    time.sleep(1.0)
                    print("✅ Chrome focused using window handle")
                    return True
                except Exception as e:
                    print(f"Window API failed: {e}")
            
            # Method 2: Alt+Tab approach with verification
            print("🔄 Using Alt+Tab method...")
            for i in range(3):
                pyautogui.hotkey('alt', 'tab')
                time.sleep(0.8)
                
                # Test if Chrome is focused by trying Ctrl+L
                try:
                    pyautogui.hotkey('ctrl', 'l')
                    time.sleep(0.3)
                    pyautogui.press('escape')
                    print("✅ Chrome focused via Alt+Tab!")
                    return True
                except:
                    continue
            
            # Method 3: Click on Chrome in taskbar area
            print("🖱️ Trying taskbar click...")
            screen_width, screen_height = pyautogui.size()
            taskbar_y = screen_height - 40
            
            # Try clicking different positions in taskbar
            for x in range(200, 800, 100):
                pyautogui.click(x, taskbar_y)
                time.sleep(0.5)
                
                # Test if Chrome is now focused
                try:
                    pyautogui.hotkey('ctrl', 'l')
                    time.sleep(0.3)
                    pyautogui.press('escape')
                    print("✅ Chrome focused via taskbar click!")
                    return True
                except:
                    continue
            
            print("⚠️ Could not focus Chrome window")
            return False
            
        except Exception as e:
            print(f"⚠️ Focus method failed: {e}")
            return False
    
    def setup_selenium_driver(self):
        """Setup Selenium WebDriver for Chrome"""
        try:
            chrome_options = Options()
            chrome_options.add_experimental_option("detach", True)
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            
            # Try to create driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.implicitly_wait(10)
            print("✅ Selenium WebDriver initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to setup Selenium: {e}")
            print("📝 Using PyAutoGUI fallback for browser control")
            return False
    
    def listen_for_commands(self):
        """Listen for voice commands continuously"""
        print("\n🎤 Listening for voice commands...")
        print("💡 Speak clearly and wait for the beep!")

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🔧 Adjusted for ambient noise")

        self.listening = True

        while self.listening:
            try:
                with self.microphone as source:
                    if not self.listening:
                        break  # if stop() was called during previous cycle

                    if self.search_mode:
                        print("\n🔍 Listening for search query...")
                        audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=10)
                    else:
                        print("\n👂 Listening...")
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                if not self.listening:
                    break  # break after finishing recognition if stopped

                print("🔄 Processing speech...")
                command = self.recognizer.recognize_google(audio).lower()
                print(f"🗣️ Heard: '{command}'")
                self.process_command(command)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("❓ Could not understand the command")
            except sr.RequestError as e:
                print(f"❌ Speech recognition error: {e}")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

        print("🛑 Voice listening stopped gracefully.")

    
    def process_command(self, command):
        """Process recognized voice command"""
        command = command.strip().lower()
        
        try:
            # Handle search mode
            if self.search_mode:
                print(f"🔍 Search query captured: '{command}'")
                self.search_query(command)
                self.search_mode = False
                return
            
            if "open chrome" in command:
                self.open_chrome()
            
            elif "search" in command and len(command.split()) == 1:
                # Just "search" command - enter search mode
                print("🎤 Search mode activated! Say your search query now...")
                self.search_mode = True
                return
            
            elif command.startswith("search ") or "search for" in command:
                # "search [query]" or "search for [query]" in one command
                if command.startswith("search for "):
                    query = command.replace("search for ", "", 1)
                else:
                    query = command.replace("search ", "", 1)
                
                if query.strip():  # Make sure query is not empty
                    self.search_query(query)
                else:
                    print("❓ No search query provided. Try: 'Search cats' or just say 'Search' first")
            
            elif "click" in command and len(command.strip()) <= 6:  # Just "click" command
                self.click_at_cursor()
            
            elif "open another tab" in command or "new tab" in command:
                self.open_new_tab()
            
            elif "close tab" in command:
                self.close_tab()
            
            elif "close browser" in command:
                self.close_browser()
            
            elif "play video" in command or "play" in command:
                self.play_video()
            
            elif "pause video" in command or "pause" in command:
                self.pause_video()
            
            elif "stop listening" in command or "exit" in command:
                self.stop_listening()
            
            else:
                print(f"❓ Unknown command: '{command}'")
                print("💡 Try: 'Open Chrome', 'Search cats', 'Play video', etc.")
                
        except Exception as e:
            print(f"❌ Error executing command '{command}': {e}")
    
    def open_chrome(self):
        """Open Chrome browser with just one tab"""
        try:
            print("🌐 Opening Chrome...")
            
            if self.chrome_path:
                # Open Chrome with Google homepage in single window
                print(f"🚀 Starting Chrome from: {self.chrome_path}")
                subprocess.Popen([
                    self.chrome_path, 
                    "--new-window", 
                    "--start-maximized",
                    "https://www.google.com"
                ])
            else:
                # Fallback - open Google in default browser
                webbrowser.open('https://www.google.com')
            
            # Wait for Chrome to fully load
            print("⏰ Waiting for Chrome to load...")
            time.sleep(4)
            
            # Try to focus the Chrome window
            self.focus_chrome_window()
            
            print("✅ Chrome opened with Google homepage!")
            
        except Exception as e:
            print(f"❌ Failed to open Chrome: {e}")
    
    def manual_chrome_focus_helper(self):
        """Helper method to manually ensure Chrome focus"""
        print("\n🎯 CHROME FOCUS HELPER:")
        print("1. Click on Chrome window manually")
        print("2. Make sure Chrome is visible on screen")
        print("3. Press any key when Chrome is focused...")
        input("Press Enter when Chrome is focused and ready...")
        
        # Test if Chrome is actually focused
        try:
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(0.3)
            pyautogui.press('escape')
            print("✅ Chrome focus confirmed!")
            return True
        except:
            print("❌ Chrome still not focused")
            return False
    
    def search_query(self, query):
        """Search for a query on Google in the current tab"""
        try:
            print(f"🔍 Searching for: '{query}'")
            
            # Try automatic focus first
            print("🎯 Attempting automatic Chrome focus...")
            focused = False
            for attempt in range(2):
                if self.focus_chrome_window():
                    focused = True
                    break
                time.sleep(1)
            
            # If automatic focus fails, offer manual option
            if not focused:
                print("⚠️ Automatic Chrome focus failed!")
                response = input("Try manual focus? (y/n): ").lower()
                if response == 'y':
                    focused = self.manual_chrome_focus_helper()
            
            if focused:
                # Use PyAutoGUI method for reliable searching
                self._search_with_pyautogui(query)
            else:
                print("❌ Cannot search - Chrome not focused!")
                print("💡 Make sure Chrome window is open and visible")
                
        except Exception as e:
            print(f"❌ Failed to search: {e}")
    
    def _search_with_pyautogui(self, query):
        """Search using PyAutoGUI - SIMPLIFIED AND RELIABLE VERSION"""
        try:
            print(f"� Searching for: '{query}'")
            
            # Step 1: Focus Chrome window (CRITICAL!)
            print("📍 Focusing Chrome...")
            focused = False
            for attempt in range(3):
                if self.focus_chrome_window():
                    focused = True
                    break
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
            
            if not focused:
                print("❌ Could not focus Chrome! Make sure Chrome is open.")
                return
            
            # Step 2: Wait and ensure Chrome is ready
            time.sleep(1.5)
            
            # Step 3: Focus address bar
            print("📍 Focusing address bar...")
            pyautogui.hotkey('ctrl', 'l')
            time.sleep(1.0)
            
            # Step 4: Clear anything in address bar
            print("📍 Clearing address bar...")
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.5)
            
            # Step 5: Type search query SLOWLY
            print(f"📍 Typing: '{query}'")
            for char in query:
                pyautogui.typewrite(char)
                time.sleep(0.08)  # Slower typing for reliability
            
            time.sleep(1.0)
            
            # Step 6: Press Enter
            print("📍 Pressing Enter...")
            pyautogui.press('enter')
            
            print("✅ Search completed!")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    def open_search_result(self, result_number):
        """Open a specific search result by number (1-5)"""
        try:
            print(f"🔗 Opening search result #{result_number}...")
            
            if self.driver:
                # Use Selenium to find and click search results
                try:
                    # Wait a moment for page to load
                    time.sleep(2)
                    
                    # Multiple selectors for different Google layouts
                    selectors = [
                        f"#search .g:nth-child({result_number}) h3 a",  # Standard Google results
                        f"#rso .g:nth-child({result_number}) h3 a",     # Alternative layout
                        f".g:nth-child({result_number}) a[href]:first-of-type",  # Backup selector
                        f".yuRUbf:nth-child({result_number}) a",        # New Google layout
                    ]
                    
                    link_clicked = False
                    for selector in selectors:
                        try:
                            links = self.driver.find_elements(By.CSS_SELECTOR, selector)
                            if links and len(links) >= 1:
                                # Get the actual result number (adjust for 0-based indexing)
                                if result_number <= len(links):
                                    link = links[0] if result_number == 1 else links[result_number - 1]
                                    self.driver.execute_script("arguments[0].click();", link)
                                    print(f"✅ Opened search result #{result_number} with Selenium!")
                                    link_clicked = True
                                    break
                        except:
                            continue
                    
                    if not link_clicked:
                        # Fallback: try to find any clickable links in results
                        try:
                            all_links = self.driver.find_elements(By.CSS_SELECTOR, "#search .g h3 a, #rso .g h3 a")
                            if all_links and result_number <= len(all_links):
                                link = all_links[result_number - 1]
                                self.driver.execute_script("arguments[0].click();", link)
                                print(f"✅ Opened search result #{result_number} (fallback method)!")
                                link_clicked = True
                        except:
                            pass
                    
                    if not link_clicked:
                        print(f"❌ Could not find search result #{result_number}")
                        return
                        
                except Exception as selenium_error:
                    print(f"⚠️ Selenium method failed: {selenium_error}")
                    # Fall back to PyAutoGUI
                    self._open_result_with_pyautogui(result_number)
            else:
                # Use PyAutoGUI fallback
                self._open_result_with_pyautogui(result_number)
                
        except Exception as e:
            print(f"❌ Failed to open search result: {e}")
    
    def _open_result_with_pyautogui(self, result_number):
        """Fallback method to open search results using PyAutoGUI"""
        try:
            print(f"🖱️ Using PyAutoGUI to open result #{result_number}...")
            
            # Scroll to top of page
            pyautogui.press('home')
            time.sleep(1)
            
            # Press Tab multiple times to navigate to search results
            # First result usually takes about 8-10 tabs, subsequent results +3 tabs each
            tab_count = 7 + (result_number - 1) * 3
            
            for i in range(tab_count):
                pyautogui.press('tab')
                time.sleep(0.1)
            
            # Press Enter to open the link
            pyautogui.press('enter')
            print(f"✅ Opened search result #{result_number} with PyAutoGUI!")
            
        except Exception as e:
            print(f"❌ PyAutoGUI fallback failed: {e}")
    
    def click_at_cursor(self):
        """Click at the current cursor position"""
        try:
            print("🖱️ Clicking at current cursor position...")
            
            # Get current cursor position
            current_x, current_y = pyautogui.position()
            print(f"📍 Cursor position: ({current_x}, {current_y})")
            
            # Click at current position
            pyautogui.click(current_x, current_y)
            print("✅ Clicked successfully!")
            
        except Exception as e:
            print(f"❌ Failed to click: {e}")
    
    def open_new_tab(self):
        """Open a new tab"""
        try:
            print("📂 Opening new tab...")
            
            if self.driver:
                # Use Selenium
                self.driver.execute_script("window.open('');")
                self.driver.switch_to.window(self.driver.window_handles[-1])
            else:
                # Use PyAutoGUI
                pyautogui.hotkey('ctrl', 't')
            
            print("✅ New tab opened!")
            
        except Exception as e:
            print(f"❌ Failed to open new tab: {e}")
    
    def close_tab(self):
        """Close current tab"""
        try:
            print("❌ Closing current tab...")
            
            if self.driver and len(self.driver.window_handles) > 1:
                # Use Selenium
                self.driver.close()
                self.driver.switch_to.window(self.driver.window_handles[-1])
            else:
                # Use PyAutoGUI
                pyautogui.hotkey('ctrl', 'w')
            
            print("✅ Tab closed!")
            
        except Exception as e:
            print(f"❌ Failed to close tab: {e}")
    
    def close_browser(self):
        """Close entire browser"""
        try:
            print("🚪 Closing browser...")
            
            if self.driver:
                self.driver.quit()
                self.driver = None
            else:
                pyautogui.hotkey('alt', 'f4')
            
            print("✅ Browser closed!")
            
        except Exception as e:
            print(f"❌ Failed to close browser: {e}")
    
    def play_video(self):
        """Play/resume video (works on YouTube and most video sites)"""
        try:
            print("▶️ Playing video...")
            
            if self.driver:
                # Try YouTube specific controls first
                try:
                    # Look for YouTube play button
                    play_button = self.driver.find_element(By.CSS_SELECTOR, ".ytp-play-button")
                    if "paused" in play_button.get_attribute("class"):
                        play_button.click()
                        print("✅ YouTube video played!")
                        return
                except NoSuchElementException:
                    pass
                
                # Try HTML5 video controls
                try:
                    video = self.driver.find_element(By.TAG_NAME, "video")
                    self.driver.execute_script("arguments[0].play();", video)
                    print("✅ HTML5 video played!")
                    return
                except NoSuchElementException:
                    pass
            
            # Fallback to spacebar (universal play/pause)
            pyautogui.press('space')
            print("✅ Video play command sent (spacebar)!")
            
        except Exception as e:
            print(f"❌ Failed to play video: {e}")

    
    
    def pause_video(self):
        """Pause video (works on YouTube and most video sites)"""
        try:
            print("⏸️ Pausing video...")
            
            if self.driver:
                # Try YouTube specific controls first
                try:
                    play_button = self.driver.find_element(By.CSS_SELECTOR, ".ytp-play-button")
                    if "paused" not in play_button.get_attribute("class"):
                        play_button.click()
                        print("✅ YouTube video paused!")
                        return
                except NoSuchElementException:
                    pass
                
                # Try HTML5 video controls
                try:
                    video = self.driver.find_element(By.TAG_NAME, "video")
                    self.driver.execute_script("arguments[0].pause();", video)
                    print("✅ HTML5 video paused!")
                    return
                except NoSuchElementException:
                    pass
            
            # Fallback to spacebar (universal play/pause)
            pyautogui.press('space')
            print("✅ Video pause command sent (spacebar)!")
            
        except Exception as e:
            print(f"❌ Failed to pause video: {e}")
    
    def stop_listening(self):
        """Stop the voice command listener"""
        print("🛑 Stopping voice command listener...")
        self.listening = False
        
        if self.driver:
            self.driver.quit()
        
        print("👋 Voice Browser Controller stopped!")
    
    def run(self):
        print("\n🚀 Starting Voice Browser Controller...")
        self.active = True
        self.listening = True   # ✅ start listening here
        try:
            self.listen_for_commands()
        except Exception as e:
            print(f"❌ Fatal error: {e}")
        finally:
            self.active = False
            print("🧩 Voice controller stopped gracefully.")

    def stop(self):
        """🛑 Gracefully stop voice recognition and browser control"""
        print("🧹 Stopping VoiceBrowserController...")
        self.listening = False  # signal stop

        try:
            # stop microphone listening if background thread exists
            if hasattr(self, "stop_listening_fn") and self.stop_listening_fn:
                    self.stop_listening_fn(wait_for_stop=False)
                    print("🎙️ Microphone listener stopped")

            if self.driver:
                self.driver.quit()
                self.driver = None
                print("✅ Browser closed cleanly")
        except Exception as e:
            print(f"⚠️ Error during stop: {e}")

        print("🔇 Voice recognition stopped.")



def main():
    """Main function to run the voice browser controller"""
    print("=" * 60)
    print("🎤 VOICE-CONTROLLED BROWSER AUTOMATION")
    print("=" * 60)
    
    # Check dependencies
    print("🔧 Checking dependencies...")
    
    try:
        import speech_recognition
        print("✅ SpeechRecognition available")
    except ImportError:
        print("❌ SpeechRecognition not found. Install with: pip install SpeechRecognition")
        return
    
    try:
        import pyautogui
        print("✅ PyAutoGUI available")
    except ImportError:
        print("❌ PyAutoGUI not found. Install with: pip install PyAutoGUI")
        return
    
    try:
        import selenium
        print("✅ Selenium available")
    except ImportError:
        print("⚠️ Selenium not found. Install with: pip install selenium")
        print("📝 Will use PyAutoGUI fallback for browser control")
    
    # Initialize and run controller
    controller = VoiceBrowserController()
    controller.run()

if __name__ == "__main__":
    main()