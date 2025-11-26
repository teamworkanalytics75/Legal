#!/usr/bin/env python3
"""
Harvard CAP JavaScript Test Runner

Use Python to run JavaScript in a headless browser to test Harvard CAP.
"""

import subprocess
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CAPJavaScriptRunner:
    """Run JavaScript tests against Harvard CAP."""

    def __init__(self):
        """Initialize the JavaScript runner."""
        self.results_dir = Path("data/case_law/1782_discovery/cap_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_test_script(self):
        """Create a Node.js script to test CAP."""
        js_code = """
const puppeteer = require('puppeteer');

async function testHarvardCAP() {
    console.log('Starting Harvard CAP Test...');

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
        await page.waitForTimeout(5000);

        // Get page info
        const title = await page.title();
        console.log('Page title:', title);

        // Look for search elements
        const searchInputs = await page.$$('input[type="search"], input[name*="search"], input[placeholder*="search"], input[placeholder*="Search"], input[placeholder*="case"], input[placeholder*="Case"]');
        console.log('Found search inputs:', searchInputs.length);

        // Look for any input elements
        const allInputs = await page.$$('input');
        console.log('Total input elements:', allInputs.length);

        // Look for buttons
        const buttons = await page.$$('button');
        console.log('Total buttons:', buttons.length);

        // Look for forms
        const forms = await page.$$('form');
        console.log('Total forms:', forms.length);

        // Test search functionality if we found search inputs
        const searchResults = {};

        if (searchInputs.length > 0) {
            console.log('Testing search functionality...');

            const queries = ['1782', '28 U.S.C. 1782', 'section 1782', 'foreign tribunal', 'Intel Corp', 'ZF Automotive'];

            for (const query of queries) {
                console.log(`\\nTesting search: "${query}"`);

                try {
                    const searchInput = searchInputs[0];

                    // Clear and type search query
                    await searchInput.click({ clickCount: 3 });
                    await searchInput.type(query);

                    // Look for search button
                    const searchButtons = await page.$$('button[type="submit"], input[type="submit"], button:contains("Search"), button:contains("Go")');

                    if (searchButtons.length > 0) {
                        console.log('Found search button, clicking...');
                        await searchButtons[0].click();
                    } else {
                        console.log('No search button found, pressing Enter...');
                        await searchInput.press('Enter');
                    }

                    // Wait for results
                    await page.waitForTimeout(5000);

                    // Look for results
                    const results = await page.$$('.result, .case, .opinion, .decision, [class*="result"], [class*="case"], [class*="search-result"]');
                    console.log(`Found ${results.length} result elements`);

                    // Get result details
                    const resultDetails = [];
                    for (let i = 0; i < Math.min(results.length, 10); i++) {
                        const result = results[i];
                        const text = await result.evaluate(el => el.textContent);
                        const link = await result.$('a');
                        const href = link ? await link.evaluate(el => el.href) : '';

                        resultDetails.push({
                            text: text.substring(0, 200),
                            href: href
                        });
                    }

                    searchResults[query] = {
                        success: true,
                        resultCount: results.length,
                        results: resultDetails
                    };

                    console.log(`Search "${query}" completed: ${results.length} results`);

                } catch (error) {
                    console.log(`Error searching for "${query}":`, error.message);
                    searchResults[query] = {
                        success: false,
                        error: error.message
                    };
                }

                // Go back to main page
                await page.goto('https://case.law', { waitUntil: 'networkidle2' });
                await page.waitForTimeout(3000);
            }
        } else {
            console.log('No search inputs found - CAP may not have search functionality');
        }

        // Save page source for analysis
        const pageSource = await page.content();

        const finalResult = {
            success: true,
            title: title,
            searchInputs: searchInputs.length,
            totalInputs: allInputs.length,
            buttons: buttons.length,
            forms: forms.length,
            pageSourceLength: pageSource.length,
            searchResults: searchResults
        };

        console.log('\\n=== FINAL RESULTS ===');
        console.log(JSON.stringify(finalResult, null, 2));

        return finalResult;

    } finally {
        await browser.close();
    }
}

// Run the test
testHarvardCAP().then(result => {
    console.log('Test completed successfully');
    process.exit(0);
}).catch(error => {
    console.error('Test failed:', error);
    process.exit(1);
});
"""

        js_file = self.results_dir / "test_cap.js"
        with open(js_file, 'w') as f:
            f.write(js_code)
        return js_file

    def install_puppeteer(self):
        """Install Puppeteer."""
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

    def run_test(self):
        """Run the JavaScript test."""
        js_file = self.create_test_script()

        try:
            logger.info("Running Harvard CAP test...")
            result = subprocess.run(['node', str(js_file)],
                                 capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                logger.info("✓ Test completed successfully")
                logger.info("Output:")
                print(result.stdout)

                # Try to parse JSON result
                try:
                    lines = result.stdout.split('\\n')
                    for line in lines:
                        if line.startswith('{') and line.endswith('}'):
                            data = json.loads(line)
                            logger.info("Parsed JSON result:")
                            print(json.dumps(data, indent=2))
                            break
                except:
                    pass

                return True
            else:
                logger.error(f"Test failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("Test timed out")
            return False
        except FileNotFoundError:
            logger.error("node not found. Please install Node.js first.")
            return False
        except Exception as e:
            logger.error(f"Error running test: {e}")
            return False

    def run(self):
        """Run the complete test."""
        logger.info("Starting Harvard CAP JavaScript Test")
        logger.info("The irony continues...")

        try:
            # Install Puppeteer
            if not self.install_puppeteer():
                logger.error("Cannot continue without Puppeteer")
                return

            # Run the test
            if self.run_test():
                logger.info("=" * 60)
                logger.info("CAP JAVASCRIPT TEST COMPLETE")
                logger.info("=" * 60)
                logger.info(f"Results saved to: {self.results_dir}")
            else:
                logger.error("JavaScript test failed")

        except Exception as e:
            logger.error(f"Error during test: {e}", exc_info=True)


def main():
    """Main entry point."""
    print("Harvard CAP JavaScript Test Runner")
    print("=" * 60)
    print("Testing Harvard's website with JavaScript automation")
    print("(The irony continues...)")
    print("=" * 60)

    runner = CAPJavaScriptRunner()
    runner.run()


if __name__ == "__main__":
    main()
