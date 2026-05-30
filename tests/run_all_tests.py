##\file run_all_tests.py
##\brief Test discovery and execution module
##\details Discovers and runs all unit tests in the tests directory
##\author Lab Team
##\version 1.0

import sys
import unittest
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

def run_all_tests():
    ##\brief Discover and run all tests
    ##\return unittest.TestResult object with test results
    
    tests_dir = Path(__file__).parent
    
    loader = unittest.TestLoader()
    
    suite = loader.discover(str(tests_dir), pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    
    result = runner.run(suite)
    
    return result


def print_summary(result):
    ##\brief Print test execution summary
    ##\param result unittest.TestResult object
    ##\return True if all tests passed, False otherwise
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    if result.wasSuccessful():
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    result = run_all_tests()
    success = print_summary(result)
    sys.exit(0 if success else 1)
