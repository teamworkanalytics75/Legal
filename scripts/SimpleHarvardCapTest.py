#!/usr/bin/env python3
"""
Simple Harvard CAP Test

Use a simple approach to test Harvard CAP without complex dependencies.
"""

import subprocess
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SimpleCAPTest:
    """Simple test of Harvard CAP using basic HTTP requests."""

    def __init__(self):
        """Initialize the test."""
        self.results_dir = Path("data/case_law/1782_discovery/cap_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def create_simple_test(self):
        """Create a simple Node.js test using built-in modules."""
        js_code = """
const https = require('https');
const http = require('http');

function makeRequest(url) {
    return new Promise((resolve, reject) => {
        const client = url.startsWith('https:') ? https : http;

        client.get(url, (res) => {
            let data = '';

            res.on('data', (chunk) => {
                data += chunk;
            });

            res.on('end', () => {
                resolve({
                    statusCode: res.statusCode,
                    headers: res.headers,
                    data: data,
                    length: data.length
                });
            });
        }).on('error', (err) => {
            reject(err);
        });
    });
}

async function testHarvardCAP() {
    console.log('Starting Harvard CAP Test...');

    const results = {};

    try {
        // Test main page
        console.log('Testing main page...');
        const mainPage = await makeRequest('https://case.law');
        results.mainPage = {
            statusCode: mainPage.statusCode,
            length: mainPage.length,
            success: mainPage.statusCode === 200
        };
        console.log(`Main page: ${mainPage.statusCode}, ${mainPage.length} chars`);

        // Test search endpoints
        const searchQueries = ['1782', '28 U.S.C. 1782', 'section 1782', 'foreign tribunal', 'Intel Corp', 'ZF Automotive'];

        for (const query of searchQueries) {
            console.log(`Testing search: "${query}"`);

            const searchUrls = [
                `https://case.law/search/?q=${encodeURIComponent(query)}`,
                `https://case.law/?q=${encodeURIComponent(query)}`,
                `https://case.law/search?q=${encodeURIComponent(query)}`
            ];

            const queryResults = {};

            for (const url of searchUrls) {
                try {
                    const response = await makeRequest(url);
                    queryResults[url] = {
                        statusCode: response.statusCode,
                        length: response.length,
                        success: response.statusCode === 200
                    };
                    console.log(`  ${url}: ${response.statusCode}, ${response.length} chars`);
                } catch (error) {
                    queryResults[url] = {
                        error: error.message,
                        success: false
                    };
                    console.log(`  ${url}: Error - ${error.message}`);
                }
            }

            results[query] = queryResults;
        }

        // Test API endpoints
        console.log('Testing API endpoints...');
        const apiEndpoints = [
            'https://case.law/api/v1/cases/',
            'https://case.law/api/search/',
            'https://case.law/api/v1/search/',
            'https://case.law/api/bulk/',
            'https://case.law/download/'
        ];

        const apiResults = {};
        for (const endpoint of apiEndpoints) {
            try {
                const response = await makeRequest(endpoint);
                apiResults[endpoint] = {
                    statusCode: response.statusCode,
                    length: response.length,
                    success: response.statusCode === 200
                };
                console.log(`  ${endpoint}: ${response.statusCode}, ${response.length} chars`);
            } catch (error) {
                apiResults[endpoint] = {
                    error: error.message,
                    success: false
                };
                console.log(`  ${endpoint}: Error - ${error.message}`);
            }
        }

        results.apiEndpoints = apiResults;

        console.log('\\n=== FINAL RESULTS ===');
        console.log(JSON.stringify(results, null, 2));

        return results;

    } catch (error) {
        console.error('Test failed:', error);
        return { error: error.message };
    }
}

// Run the test
testHarvardCAP().then(result => {
    console.log('Test completed');
    process.exit(0);
}).catch(error => {
    console.error('Test failed:', error);
    process.exit(1);
});
"""

        js_file = self.results_dir / "simple_cap_test.js"
        with open(js_file, 'w') as f:
            f.write(js_code)
        return js_file

    def run_test(self):
        """Run the simple test."""
        js_file = self.create_simple_test()

        try:
            logger.info("Running simple Harvard CAP test...")
            result = subprocess.run([
                'C:\\Program Files\\nodejs\\node.exe',
                str(js_file)
            ], capture_output=True, text=True, timeout=60)

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
        except Exception as e:
            logger.error(f"Error running test: {e}")
            return False

    def run(self):
        """Run the complete test."""
        logger.info("Starting Simple Harvard CAP Test")
        logger.info("The irony continues...")

        try:
            if self.run_test():
                logger.info("=" * 60)
                logger.info("SIMPLE CAP TEST COMPLETE")
                logger.info("=" * 60)
                logger.info(f"Results saved to: {self.results_dir}")
            else:
                logger.error("Simple test failed")

        except Exception as e:
            logger.error(f"Error during test: {e}", exc_info=True)


def main():
    """Main entry point."""
    print("Simple Harvard CAP Test")
    print("=" * 60)
    print("Testing Harvard's website with simple HTTP requests")
    print("(The irony continues...)")
    print("=" * 60)

    tester = SimpleCAPTest()
    tester.run()


if __name__ == "__main__":
    main()
