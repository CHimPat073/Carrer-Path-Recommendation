"""
Test file to verify CareerPilot-AI is working correctly.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all main modules can be imported."""
    print("Testing imports...")

    try:
        from backend.app.main import app
        print("  [OK] FastAPI app imported successfully")
    except ImportError as e:
        print(f"  [FAIL] Failed to import app: {e}")
        return False

    try:
        from ml.preprocessing import data_loader, cleaning, role_normalizer
        print("  [OK] ML preprocessing modules imported successfully")
    except ImportError as e:
        print(f"  [FAIL] Failed to import ML modules: {e}")
        return False

    return True


def test_app_config():
    """Test FastAPI app configuration."""
    print("\nTesting app configuration...")

    from backend.app.main import app

    # Check app title
    assert app.title == "CareerPilot-AI API", "App title mismatch"
    print(f"  [OK] App title: {app.title}")

    # Check routes
    routes = [route.path for route in app.routes]
    print(f"  [OK] Available routes: {routes}")

    return True


def test_root_endpoint():
    """Test the root endpoint."""
    print("\nTesting root endpoint...")

    from backend.app.main import app, read_root

    result = read_root()
    assert result == {"message": "CareerPilot-AI backend is running"}, "Root endpoint response mismatch"
    print(f"  [OK] Root endpoint response: {result}")

    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("CareerPilot-AI Test Suite")
    print("=" * 50)

    tests = [
        test_imports,
        test_app_config,
        test_root_endpoint,
    ]

    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"  [FAIL] Test failed with error: {e}")
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("SUCCESS: All tests passed!")
    else:
        print("FAILURE: Some tests failed!")
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)