import asyncio
import subprocess
import os
import re
from pathlib import Path
from playwright.async_api import async_playwright

page_waittimeout = 5000
scroll_time = 5
remote_debugging_port = 9222

async def interact_with_twitter(tweet_url):
    # Get home directory and set up Chrome profile path
    home = os.path.expanduser("~")
    chrome_profile_dir = f"{home}/.local/share/news_agent/chrome_profile"
    
    # Create the directory if it doesn't exist
    Path(chrome_profile_dir).mkdir(parents=True, exist_ok=True)
    
    chrome_exe = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    
    # Launch Chrome manually with CDP (like the Rust code does)
    print("🚀 Launching Chrome manually with CDP...")
    chrome_process = subprocess.Popen([
        chrome_exe,
        f"--remote-debugging-port={remote_debugging_port}",
        f"--user-data-dir={chrome_profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--enable-automation"
    ])
    
    try:
        # Wait a moment for Chrome to start
        await asyncio.sleep(2)
        
        # Connect to the existing Chrome instance via CDP (Option 1: direct connection)
        print("Connecting to chrome")
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(f"http://127.0.0.1:{remote_debugging_port}")
            
            try:
                # Get existing context or create new page
                contexts = browser.contexts
                if contexts and contexts[0].pages:
                    page = contexts[0].pages[0]
                else:
                    page = await browser.new_page()
                
                # Go to the target tweet URL
                print(f"visiting {tweet_url}")
                await page.goto(tweet_url)
                
                # Wait for the page to load (adjust timeout if necessary)
                await page.wait_for_selector('article', timeout=page_waittimeout)
                
                # Expand the tweet's text (click on "Show More" button if exists)
                try:
                    await page.click('div[data-testid="tweet"] button[aria-label="Show more"]', timeout=3000)
                    print("Expanded tweet text")
                except:
                    print("No 'Show More' button found")
                
                # Scroll to load replies or more content
                last_height = await page.evaluate('document.body.scrollHeight')
                for _ in range(scroll_time):  # Scroll 5 times (you can adjust this)
                    await page.evaluate('window.scrollTo(0, document.body.scrollHeight);')
                    await page.wait_for_timeout(2000)  # Wait for 2 seconds for new content to load
                    new_height = await page.evaluate('document.body.scrollHeight')
                    if new_height == last_height:  # Stop scrolling if no new content is loaded
                        break
                    last_height = new_height
                
                # Optionally, extract all tweet text and links
                tweets = await page.query_selector_all('article div[data-testid="tweetText"]')
                tweet_texts = [await tweet.inner_text() for tweet in tweets]
                
                links = await page.query_selector_all('a')
                link_urls = []
                # Regex pattern: matches URLs starting with http/https, host is arxiv.org or huggingface.co/papers/,
                # followed by path containing arxiv ID format (4 digits.5 digits, e.g., 2512.23705)
                paper_url_pattern = r'https?://(?:arxiv\.org|huggingface\.co/papers)/[^\s]*\d{4}\.\d{5}'
                
                for link in links:
                    href = await link.get_attribute('href')
                    if not href or "http" not in href:
                        continue
                    
                    # Handle t.co shortened links
                    if "t.co" in href:
                        print(f"Found t.co link {href}")
                        # Twitter often shows the actual destination in the link text, but it may be
                        # split across multiple spans with truncation ("...")
                        # Use textContent which automatically concatenates all text nodes (handles split spans)
                        link_text = await link.evaluate('el => el.textContent')
                        print(f"Link text (textContent): {link_text}")
                        
                        # Remove truncation indicators and whitespace that might break the pattern
                        cleaned_text = link_text.replace('...', '').replace(' ', '') if link_text else ''
                        print(f"Cleaned text: {cleaned_text}")
                        
                        # Extract matching URLs from the cleaned text
                        matches = re.findall(paper_url_pattern, cleaned_text)
                        if matches:
                            print(f"Extracted URLs: {matches}")
                            link_urls.extend(matches)
                            continue
                        
                        # If pattern didn't match, try to reconstruct from partial match
                        # Look for pattern like "huggingface.co/papers/2512.23" + "705" = "2512.23705"
                        partial_url_pattern = r'(https?://)?(arxiv\.org|huggingface\.co/papers)/[^\s]*?(\d{4}\.\d{2,4})'
                        partial_match = re.search(partial_url_pattern, cleaned_text)
                        if partial_match:
                            protocol = partial_match.group(1) or 'https://'
                            host = partial_match.group(2)
                            partial_id = partial_match.group(3)  # e.g., "2512.23"
                            
                            # Look for continuation digits after the partial ID
                            id_parts = partial_id.split('.')
                            if len(id_parts) == 2 and len(id_parts[1]) < 5:
                                # Find the rest of the ID in the text
                                continuation_pattern = rf'{re.escape(partial_id)}(\d+)'
                                continuation_match = re.search(continuation_pattern, cleaned_text)
                                if continuation_match:
                                    remaining_digits = continuation_match.group(1)
                                    full_id = f"{id_parts[0]}.{id_parts[1]}{remaining_digits}"
                                    # Verify it matches the full pattern (4 digits.5 digits)
                                    if re.match(r'\d{4}\.\d{5}', full_id):
                                        reconstructed_url = f"{protocol}{host}/{full_id}"
                                        print(f"Reconstructed URL from partial match: {reconstructed_url}")
                                        link_urls.append(reconstructed_url)
                                        continue
                        
                        # # If visible text doesn't have the URL, follow the redirect
                        # try:
                        #     print(f"Opening link {href} in a new page")
                        #     # Open link in a new page to get the final URL
                        #     new_page = await browser.new_page()
                        #     response = await new_page.goto(href, wait_until='domcontentloaded', timeout=5000)
                        #     if response:
                        #         final_url = response.url
                        #         if re.search(paper_url_pattern, final_url):
                        #             link_urls.append(final_url)
                        #     await new_page.close()
                        # except Exception as e:
                        #     print(f"Failed to resolve t.co link {href}: {e}")
                        #     # Fallback: use the t.co link itself if we can't resolve it
                        #     # (though it won't match our filter, so we skip it)
                        # continue
                    
                    # For non-t.co links, check if they match our criteria using the regex pattern
                    if re.search(paper_url_pattern, href):
                        link_urls.append(href)
                
                # Deduplicate URLs while preserving order
                link_urls = list(dict.fromkeys(link_urls))
                
                # Print all extracted content
                print("Extracted tweets:")
                for text in tweet_texts:
                    print(text)
                
                print("\nExtracted links:")
                for link in link_urls:
                    print(link)
                
                # Capture screenshot of the whole page after unfolding and scrolling
                # await page.screenshot(path='tweet_screenshot.png')
                # print("Screenshot saved as tweet_screenshot.png")
                
            finally:
                # Disconnect from Playwright CDP connection
                # The context manager will handle cleanup, but we can also explicitly close
                await browser.close()
                print("Disconnected from Chrome CDP")
    finally:
        # Kill the Chrome process we started
        if chrome_process.poll() is None:  # Process is still running
            print("Terminating Chrome process...")
            chrome_process.terminate()
            # Wait a bit for graceful shutdown, then force kill if needed
            try:
                chrome_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                chrome_process.kill()
                chrome_process.wait()
            print("Chrome process terminated")
        

# Run the async function
if __name__ == "__main__":
    tweet_url = 'https://x.com/HuggingPapers/status/2007245681388294302'
    asyncio.run(interact_with_twitter(tweet_url))