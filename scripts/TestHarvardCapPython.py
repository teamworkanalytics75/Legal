#!/usr/bin/env python3
"""
Harvard CAP Python Test

Use Python with requests-html to test Harvard CAP website
with JavaScript execution capabilities.
"""

import asyncio
import json
import logging
from pathlib import Path
from requests_html import AsyncHTMLSession

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CAPPythonTest:
    """Test Harvard CAP using Python with JavaScript support."""

    def __init__(self):
        """Initialize the test."""
        self.results_dir = Path("data/case_law/1782_discovery/cap_test")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.session = AsyncHTMLSession()
        self.results = {}

    async def test_main_page(self):
        """Test the main page."""
        logger.info("Testing main page...")

        try:
            r = await self.session.get('https://case.law')
            await r.html.arender(timeout=20)

            title = r.html.find('title', first=True)
            title_text = title.text if title else "No title found"

            logger.info(f"Page title: {title_text}")

            # Look for search elements
            search_inputs = r.html.find('input[type="search"], input[name*="search"], input[placeholder*="search"], input[placeholder*="Search"]')
            logger.info(f"Found {len(search_inputs)} search inputs")

            # Look for any input elements
            all_inputs = r.html.find('input')
            logger.info(f"Found {len(all_inputs)} total input elements")

            # Look for buttons
            buttons = r.html.find('button')
            logger.info(f"Found {len(buttons)} buttons")

            # Look for forms
            forms = r.html.find('form')
            logger.info(f"Found {len(forms)} forms")

            # Save page source
            with open(self.results_dir / "main_page_source.html", 'w', encoding='utf-8') as f:
                f.write(r.html.html)

            return {
                'success': True,
                'title': title_text,
                'search_inputs': len(search_inputs),
                'total_inputs': len(all_inputs),
                'buttons': len(buttons),
                'forms': len(forms),
                'page_source_length': len(r.html.html)
            }

        except Exception as e:
            logger.error(f"Error testing main page: {e}")
            return {'success': False, 'error': str(e)}

    async def test_search_functionality(self):
        """Test search functionality."""
        logger.info("Testing search functionality...")

        search_queries = ['1782', '28 U.S.C. 1782', 'section 1782', 'foreign tribunal', 'Intel Corp', 'ZF Automotive']
        search_results = {}

        for query in search_queries:
            logger.info(f"Testing search: '{query}'")

            try:
                # Try different search URL patterns
                search_urls = [
                    f'https://case.law/search/?q={query}',
                    f'https://case.law/?q={query}',
                    f'https://case.law/search?q={query}',
                ]

                query_results = {}

                for url in search_urls:
                    try:
                        r = await self.session.get(url)
                        await r.html.arender(timeout=20)

                        # Look for results
                        results = r.html.find('.result, .case, .opinion, .decision, [class*="result"], [class*="case"]')

                        query_results[url] = {
                            'success': True,
                            'result_count': len(results),
                            'page_source_length': len(r.html.html)
                        }

                        logger.info(f"  {url}: {len(results)} results")

                        # Get result details
                        result_details = []
                        for i, result in enumerate(results[:10]):
                            text = result.text[:200] if result.text else ''
                            link = result.find('a', first=True)
                            href = link.attrs.get('href', '') if link else ''

                            result_details.append({
                                'text': text,
                                'href': href
                            })

                        query_results[url]['results'] = result_details

                    except Exception as e:
                        query_results[url] = {
                            'success': False,
                            'error': str(e)
                        }
                        logger.info(f"  {url}: Error - {e}")

                search_results[query] = query_results

            except Exception as e:
                logger.error(f"Error testing search '{query}': {e}")
                search_results[query] = {'error': str(e)}

        return search_results

    async def test_case_access(self):
        """Test accessing specific cases."""
        logger.info("Testing case access...")

        # Try to access the main page first
        try:
            r = await self.session.get('https://case.law')
            await r.html.arender(timeout=20)

            # Look for any case links
            case_links = r.html.find('a[href*="case"], a[href*="opinion"], a[href*="decision"]')
            logger.info(f"Found {len(case_links)} potential case links")

            case_results = {}

            # Test first few links
            for i, link in enumerate(case_links[:5]):
                try:
                    href = link.attrs.get('href', '')
                    text = link.text[:100] if link.text else ''

                    logger.info(f"Testing case link {i+1}: {text}...")

                    # Try to access the case
                    if href.startswith('/'):
                        href = f'https://case.law{href}'

                    case_r = await self.session.get(href)
                    await case_r.html.arender(timeout=20)

                    # Look for case content
                    case_content = case_r.html.find('.case-content, .opinion-text, .decision-text, .case-text')

                    case_results[f'case_{i+1}'] = {
                        'success': True,
                        'href': href,
                        'text': text,
                        'case_content_elements': len(case_content),
                        'page_title': case_r.html.find('title', first=True).text if case_r.html.find('title', first=True) else 'No title'
                    }

                    logger.info(f"  Case {i+1}: {len(case_content)} content elements")

                except Exception as e:
                    logger.error(f"Error testing case link {i+1}: {e}")
                    case_results[f'case_{i+1}'] = {'error': str(e)}

            return case_results

        except Exception as e:
            logger.error(f"Error testing case access: {e}")
            return {'error': str(e)}

    async def run_all_tests(self):
        """Run all tests."""
        logger.info("Starting Harvard CAP Python Test")
        logger.info("The irony continues...")

        try:
            # Test main page
            main_page_result = await self.test_main_page()
            self.results['main_page'] = main_page_result

            # Test search functionality
            search_results = await self.test_search_functionality()
            self.results['search'] = search_results

            # Test case access
            case_results = await self.test_case_access()
            self.results['case_access'] = case_results

            # Save results
            with open(self.results_dir / "python_test_results.json", 'w') as f:
                json.dump(self.results, f, indent=2)

            # Analyze results
            self.analyze_results()

            logger.info("=" * 60)
            logger.info("CAP PYTHON TEST COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {self.results_dir}")

        except Exception as e:
            logger.error(f"Error during test: {e}", exc_info=True)
        finally:
            await self.session.close()

    def analyze_results(self):
        """Analyze the test results."""
        logger.info("\n" + "=" * 60)
        logger.info("CAP Python Test Analysis")
        logger.info("=" * 60)

        # Analyze main page
        main_page = self.results.get('main_page', {})
        if main_page.get('success', False):
            logger.info(f"✓ Main page accessible: {main_page.get('title', 'Unknown')}")
            logger.info(f"  Search inputs: {main_page.get('search_inputs', 0)}")
            logger.info(f"  Total inputs: {main_page.get('total_inputs', 0)}")
            logger.info(f"  Buttons: {main_page.get('buttons', 0)}")
            logger.info(f"  Forms: {main_page.get('forms', 0)}")
        else:
            logger.info(f"✗ Main page failed: {main_page.get('error', 'Unknown error')}")

        # Analyze search results
        search_results = self.results.get('search', {})
        successful_searches = 0
        total_results = 0

        for query, results in search_results.items():
            logger.info(f"\nSearch: '{query}'")

            has_success = False
            for url, data in results.items():
                if data.get('success', False):
                    has_success = True
                    result_count = data.get('result_count', 0)
                    total_results += result_count
                    logger.info(f"  ✓ {url}: {result_count} results")
                else:
                    logger.info(f"  ✗ {url}: {data.get('error', 'Unknown error')}")

            if not has_success:
                logger.info(f"  ✗ No successful searches")
            else:
                successful_searches += 1

        # Analyze case access
        case_results = self.results.get('case_access', {})
        successful_cases = 0
        for case_id, result in case_results.items():
            if result.get('success', False):
                successful_cases += 1
                logger.info(f"✓ {case_id}: {result.get('case_content_elements', 0)} content elements")
            else:
                logger.info(f"✗ {case_id}: {result.get('error', 'Unknown error')}")

        logger.info(f"\nSUMMARY:")
        logger.info(f"Successful searches: {successful_searches}")
        logger.info(f"Total results found: {total_results}")
        logger.info(f"Successful case access: {successful_cases}")

        if successful_searches > 0 or successful_cases > 0:
            logger.info("✓ CAP appears to be accessible and functional")
            logger.info("✓ We can potentially extract §1782 cases from CAP")
        else:
            logger.info("✗ CAP may not be accessible or may not have the cases we need")
            logger.info("✗ May need to try different approach or give up on CAP")


async def main():
    """Main entry point."""
    print("Harvard CAP Python Test")
    print("=" * 60)
    print("Testing Harvard's website with Python + JavaScript")
    print("(The irony continues...)")
    print("=" * 60)

    tester = CAPPythonTest()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
