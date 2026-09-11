import unittest
from pathlib import Path
import rbf,rbf_r02
class RevisionTests(unittest.TestCase):
    def test_only_penalty_strength_changes(self):
        p=Path(__file__).resolve().parent
        self.assertEqual((p/'rbf.py').read_text().replace('lambda_value=.01,','lambda_value=1.,'),(p/'rbf_r02.py').read_text())
        before=rbf.CONFIG.copy();after=rbf_r02.CONFIG.copy()
        self.assertEqual(before.pop('lambda_value'),.01);self.assertEqual(after.pop('lambda_value'),1.)
        self.assertEqual(before,after)
if __name__=='__main__':unittest.main()
