#!/usr/bin/env python3
"""
Harvard CAP JavaScript Automation

Use JavaScript directly to interact with the Harvard CAP website
instead of browser automation. Much simpler and more efficient.
"""

import subprocess
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CAPJavaScriptAutomation:
    """Use JavaScript to automate Harvard CAP interactions."""

    def __init__(self):
        """Initialize the JavaScript automation."""
        self.results_dir = Path("data/case_law/1782_discovery/cap_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # JavaScript code to interact with CAP
        self.js_code = """
        // JavaScript code to interact with Harvard CAP
        const puppeteer = require('puppeteer');

        async function testCAP() {
            const browser = await puppeteer.launch({
                headless: true,
                args: ['--no-sandbox', '--disable-setuid-sandbox']
            });

            try {
                const page = await browser.newPage();
                await page.setViewport({ width: 1920, height: 1080 });

                console.log('Navigating to case.law...');
                await page.goto('https://case.law', { waitUntil: 'networkidle2' });

                // Wait for page to fully load
                await page.waitForTimeout(3000);

                // Get page title
                const title = await page.title();
                console.log('Page title:', title);

                // Look for search elements
                const searchInputs = await page.$$('input[type="search"], input[name*="search"], input[placeholder*="search"], input[placeholder*="Search"]');
                console.log('Found search inputs:', searchInputs.length);

                // Look for any input elements
                const allInputs = await page.$$('input');
                console.log('Total input elements:', allInputs.length);

                // Try to find and interact with search
                if (searchInputs.length > 0) {
                    const searchInput = searchInputs[0];

                    // Test search queries
                    const queries = ['1782', '28 U.S.C. 1782', 'section 1782', 'foreign tribunal', 'Intel Corp', 'ZF Automotive'];

                    for (const query of queries) {
                        console.log(`Testing search: ${query}`);

                        try {
                            // Clear and type search query
                            await searchInput.click({ clickCount: 3 });
                            await searchInput.type(query);

                            // Look for search button
                            const searchButtons = await page.$$('button[type="submit"], input[type="submit"], button:contains("Search")');

                            if (searchButtons.length > 0) {
                                await searchButtons[0].click();
                            } else {
                                // Try pressing Enter
                                await searchInput.press('Enter');
                            }

                            // Wait for results
                            await page.waitForTimeout(3000);

                            // Look for results
                            const results = await page.$$('.result, .case, .opinion, .decision, [class*="result"], [class*="case"]');
                            console.log(`Found ${results.length} result elements for "${query}"`);

                            // Get result details
                            const resultDetails = [];
                            for (let i = 0; i < Math.min(results.length, 10); i++) {
                                const result = results[i];
                                const text = await result.evaluate(el => el.textContent);
                                const href = await result.evaluate(el => el.href || el.querySelector('a')?.href);

                                resultDetails.push({
                                    text: text.substring(0, 200),
                                    href: href
                                });
                            }

                            console.log('Result details:', JSON.stringify(resultDetails, null, 2));

                        } catch (error) {
                            console.log(`Error searching for "${query}":`, error.message);
                        }

                        // Go back to main page
                        await page.goto('https://case.law', { waitUntil: 'networkidle2' });
                        await page.waitForTimeout(2000);
                    }
                }

                // Save page source for analysis
                const pageSource = await page.content();
                console.log('Page source length:', pageSource.length);

                return {
                    success: true,
                    title: title,
                    searchInputs: searchInputs.length,
                    totalInputs: allInputs.length,
                    pageSourceLength: pageSource.length
                };

            } finally {
                await browser.close();
            }
        }

        // Run the test
        testCAP().then(result => {
            console.log('Final result:', JSON.stringify(result, null, 2));
        }).catch(error => {
            console.error('Error:', error);
        });
        """

    def create_js_file(self):
        """Create the JavaScript file."""
        js_file = self.results_dir / "test_cap.js"
        with open(js_file, 'w') as f:
            f.write(self.js_code)
        return js_file

    def install_puppeteer(self):
        """Install Puppeteer if not already installed."""
        try:
            logger.info("Installing Puppeteer...")
            result = subprocess.run(['npm', 'install', 'puppeteer'],
                                 capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info("✓ Puppeteer installed successfully")
                return True
            else:
                logger.error(f"Failed to install Puppeteer: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Puppeteer installation timed out")
            return False
        except FileNotFoundError:
            logger.error("npm not found. Please install Node.js first.")
            return False
        except Exception as e:
            logger.error(f"Error installing Puppeteer: {e}")
            return False

    def run_js_test(self):
        """Run the JavaScript test."""
        js_file = self.create_js_file()

        try:
            logger.info("Running JavaScript test...")
            result = subprocess.run(['node', str(js_file)],
                                 capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✓ JavaScript test completed successfully")
                logger.info("Output:")
                print(result.stdout)
                return True
            else:
                logger.error(f"JavaScript test failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("JavaScript test timed out")
            return False
        except FileNotFoundError:
            logger.error("node not found. Please install Node.js first.")
            return False
        except Exception as e:
            logger.error(f"Error running JavaScript test: {e}")
            return False

    def run(self):
        """Run the JavaScript automation."""
        logger.info("Starting Harvard CAP JavaScript Automation")
        logger.info("The irony continues...")

        try:
            # Install Puppeteer
            if not self.install_puppeteer():
                logger.error("Cannot continue without Puppeteer")
                return

            # Run the test
            if self.run_js_test():
                logger.info("=" * 60)
                logger.info("CAP JAVASCRIPT AUTOMATION COMPLETE")
                logger.info("=" * 60)
                logger.info(f"Results saved to: {self.results_dir}")
            else:
                logger.error("JavaScript automation failed")

        except Exception as e:
            logger.error(f"Error during JavaScript automation: {e}", exc_info=True)


def main():
    """Main entry point."""
    print("Harvard CAP JavaScript Automation")
    print("=" * 60)
    print("Testing Harvard's website with JavaScript")
    print("(The irony continues...)")
    print("=" * 60)

    automation = CAPJavaScriptAutomation()
    automation.run()


if __name__ == "__main__":
    main()
