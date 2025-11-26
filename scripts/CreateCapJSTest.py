#!/usr/bin/env python3
"""
Harvard CAP Simple JavaScript Test

Create JavaScript code that can be run directly in a browser console
to test Harvard CAP's search functionality.
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CAPSimpleJSTest:
    """Create simple JavaScript code for browser console testing."""

    def __init__(self):
        """Initialize the simple JS test."""
        self.results_dir = Path("data/case_law/1782_discovery/cap_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_browser_console_script(self):
        """Create JavaScript code to run in browser console."""
        js_code = """
// Harvard CAP Browser Console Test Script
// Copy and paste this into your browser console on case.law

console.log('Starting Harvard CAP Test...');

// Test function to search for cases
async function testCAPSearch(query) {
    console.log(`Testing search: "${query}"`);

    try {
        // Look for search input
        const searchInputs = document.querySelectorAll('input[type="search"], input[name*="search"], input[placeholder*="search"], input[placeholder*="Search"]');
        console.log(`Found ${searchInputs.length} search inputs`);

        if (searchInputs.length === 0) {
            console.log('No search inputs found');
            return { success: false, error: 'No search inputs found' };
        }

        const searchInput = searchInputs[0];

        // Clear and type search query
        searchInput.focus();
        searchInput.value = '';
        searchInput.value = query;

        // Trigger input event
        searchInput.dispatchEvent(new Event('input', { bubbles: true }));

        // Look for search button
        const searchButtons = document.querySelectorAll('button[type="submit"], input[type="submit"], button:contains("Search")');
        console.log(`Found ${searchButtons.length} search buttons`);

        if (searchButtons.length > 0) {
            searchButtons[0].click();
        } else {
            // Try pressing Enter
            searchInput.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
        }

        // Wait for results
        await new Promise(resolve => setTimeout(resolve, 3000));

        // Look for results
        const results = document.querySelectorAll('.result, .case, .opinion, .decision, [class*="result"], [class*="case"]');
        console.log(`Found ${results.length} result elements`);

        // Get result details
        const resultDetails = [];
        for (let i = 0; i < Math.min(results.length, 10); i++) {
            const result = results[i];
            const text = result.textContent || '';
            const link = result.querySelector('a');
            const href = link ? link.href : '';

            resultDetails.push({
                text: text.substring(0, 200),
                href: href
            });
        }

        return {
            success: true,
            query: query,
            resultCount: results.length,
            results: resultDetails
        };

    } catch (error) {
        console.error(`Error searching for "${query}":`, error);
        return { success: false, error: error.message };
    }
}

// Test multiple queries
async function runAllTests() {
    const queries = ['1782', '28 U.S.C. 1782', 'section 1782', 'foreign tribunal', 'Intel Corp', 'ZF Automotive'];
    const results = {};

    for (const query of queries) {
        console.log(`\\n--- Testing: ${query} ---`);
        results[query] = await testCAPSearch(query);

        // Wait between searches
        await new Promise(resolve => setTimeout(resolve, 2000));
    }

    console.log('\\n=== FINAL RESULTS ===');
    console.log(JSON.stringify(results, null, 2));

    return results;
}

// Run the tests
runAllTests().then(results => {
    console.log('All tests completed');
}).catch(error => {
    console.error('Test failed:', error);
});

// Also provide individual test functions
window.testCAPSearch = testCAPSearch;
window.runAllTests = runAllTests;

console.log('Test functions available: testCAPSearch(query), runAllTests()');
"""

        return js_code

    def create_instructions(self):
        """Create instructions for running the test."""
        instructions = """
# Harvard CAP Browser Console Test Instructions

## How to Run This Test

1. **Open your browser** and go to https://case.law

2. **Open Developer Tools**:
   - Chrome/Edge: Press F12 or Ctrl+Shift+I
   - Firefox: Press F12 or Ctrl+Shift+K
   - Safari: Press Cmd+Option+I

3. **Go to Console tab**

4. **Copy and paste the JavaScript code** from the file below

5. **Press Enter** to run the code

6. **Watch the results** in the console

## What the Test Does

- Tests search functionality for §1782 cases
- Searches for: "1782", "28 U.S.C. 1782", "section 1782", "foreign tribunal", "Intel Corp", "ZF Automotive"
- Reports number of results found
- Shows details of first 10 results

## Expected Results

If CAP is working:
- Should find search inputs
- Should be able to perform searches
- Should return results for some queries

If CAP is not working:
- No search inputs found
- Searches return no results
- Errors in console

## Manual Test Functions

After running the main script, you can also test individual queries:

```javascript
// Test a single query
testCAPSearch("1782");

// Run all tests again
runAllTests();
```

## Troubleshooting

- If you get errors about CORS or security, try disabling browser security
- If searches return no results, CAP may not have the cases we need
- If no search inputs are found, CAP may have changed their interface
"""

        return instructions

    def run(self):
        """Create the test files."""
        logger.info("Creating Harvard CAP Browser Console Test")
        logger.info("The irony continues...")

        try:
            # Create JavaScript file
            js_code = self.create_browser_console_script()
            js_file = self.results_dir / "cap_browser_console_test.js"
            with open(js_file, 'w') as f:
                f.write(js_code)

            # Create instructions file
            instructions = self.create_instructions()
            instructions_file = self.results_dir / "CAP_TEST_INSTRUCTIONS.md"
            with open(instructions_file, 'w') as f:
                f.write(instructions)

            logger.info("=" * 60)
            logger.info("CAP BROWSER CONSOLE TEST CREATED")
            logger.info("=" * 60)
            logger.info(f"JavaScript file: {js_file}")
            logger.info(f"Instructions: {instructions_file}")
            logger.info("")
            logger.info("To run the test:")
            logger.info("1. Go to https://case.law")
            logger.info("2. Open browser console (F12)")
            logger.info("3. Copy/paste the JavaScript code")
            logger.info("4. Press Enter to run")
            logger.info("")
            logger.info("The irony of using Harvard's database to sue Harvard continues...")

        except Exception as e:
            logger.error(f"Error creating test files: {e}", exc_info=True)


def main():
    """Main entry point."""
    print("Harvard CAP Browser Console Test Creator")
    print("=" * 60)
    print("Creating JavaScript code to test Harvard's website")
    print("(The irony continues...)")
    print("=" * 60)

    test_creator = CAPSimpleJSTest()
    test_creator.run()


if __name__ == "__main__":
    main()
