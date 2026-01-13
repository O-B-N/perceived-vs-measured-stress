import unittest
import pandas as pd
from clean_student_data import clean_data_logic


class TestStudentCleaning(unittest.TestCase):
    """
    Test suite to verify that our data cleaning works as expected.
    """

    def test_valid_data_stays(self):
        """
        Verify that rows with perfect data (M, F, 0-10) are not deleted.
        """
        # Create "fake" valid data for testing.
        data = {
            'Gender': ['M', 'F'],
            'Stress_Level_Biosensor': [5, 8],
            'Stress_Level_Self_Report': [4, 7]
        }
        df = pd.DataFrame(data)
        
        # Run the cleaning logic.
        result = clean_data_logic(df)
        
        # We expect 2 rows to remain.
        self.assertEqual(len(result), 2, "Should keep both valid rows.")

    def test_case_insensitivity_gender(self):
        """
        Check that BOTH uppercase (M, F) and lowercase (m, f) are accepted.
        """
        data = {
            'Gender': ['m', 'f', 'M', 'F'], 
            'Stress_Level_Biosensor': [5, 5, 5, 5],
            'Stress_Level_Self_Report': [5, 5, 5, 5]
        }
        df = pd.DataFrame(data)
        
        # Run the cleaning logic.
        result = clean_data_logic(df)
        
        # We expect all 4 rows to stay because our code is now "smart" enough.
        self.assertEqual(len(result), 4, "Lowercase m, f and Uppercase M, F should all be accepted.")

    def test_out_of_range_stress(self):
        """
        Ensure stress levels outside 0-10 (like 99) are removed.
        """
        data = {
            'Gender': ['M', 'M'],
            'Stress_Level_Biosensor': [5, 99],  # 99 is a logical error.
            'Stress_Level_Self_Report': [5, 5]
        }
        df = pd.DataFrame(data)
        
        result = clean_data_logic(df)
        
        # Only 1 row (the first one) should remain.
        self.assertEqual(len(result), 1, "Rows with out-of-range stress levels must be removed.")
        # Verify that 99 is gone.
        self.assertNotIn(99, result['Stress_Level_Biosensor'].values)

    def test_invalid_gender(self):
        """
        Ensure random text in Gender (like 'Other') is filtered out.
        """
        data = {
            'Gender': ['M', 'Non-Binary', 'Alien'],
            'Stress_Level_Biosensor': [5, 5, 5],
            'Stress_Level_Self_Report': [5, 5, 5]
        }
        df = pd.DataFrame(data)
        
        result = clean_data_logic(df)
        
        # Only 'M' should remain (1 row)
        self.assertEqual(len(result), 1, "Only allowed gender categories (M/F) should remain.")


if __name__ == '__main__':
    # Start the testing process
    unittest.main()
    