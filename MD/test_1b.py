# Import the MonteCarlo class
from MonteCarlo import MonteCarlo

# Define a function that try to call the *__init__()* method
def test_init_method():
    try:
        MonteCarlo().__init__()  # Call the method
        print("Method call succeeded")
    except Exception as e:
        print(f"Method call raised an error: {e}")

# In the script is launched with Python, call Pytest
if __name__ == "__main__":
    import pytest
    pytest.main(["-s", __file__])