import unittest,math
from tools.charades_prototype import LABEL_NAMES,POLICY,mapping,prompt,interface,choose,participate

class FullCharadesTests(unittest.TestCase):
    def test_vocabulary_and_prompt_are_charades(self):
        p=prompt('a');self.assertIn('SIT',p);self.assertIn('LAUGH',p);self.assertNotIn('POUR',p)
        self.assertEqual(set(mapping('a')),set(LABEL_NAMES));self.assertEqual(mapping('a'),mapping('a'))
    def test_kitchen_default_not_modified(self):
        from tools.qwen_source_interface import LABELS
        interface();self.assertIn('POUR',LABELS);self.assertNotIn('LAUGH',LABELS)
    def test_digit_to_class_index_all_ten(self):
        reader=interface();names=mapping('x')
        for index,name in enumerate(names):
            logits=[-10.]*10;logits[index]=10.
            r=reader.normalize_digit_logits(logits,names)
            self.assertEqual(r['class_id'],LABEL_NAMES.index(name));self.assertAlmostEqual(sum(r['class_probabilities']),1.)
    def test_fixed_budget_does_not_use_score(self):
        rows=[dict(candidate_id=str(i),video_id=str(i%10),absolute_feature_index=i,score=i) for i in range(400)]
        first=[r['candidate_id'] for r in choose(rows)]
        for r in rows:r['score']=-r['score']
        self.assertEqual(first,[r['candidate_id'] for r in choose(rows)]);self.assertEqual(len(first),200)
    def test_top40_ties_excluded(self):
        rows=[dict(candidate_id=str(i),video_id=str(i),absolute_feature_index=i,class_probabilities=[.9-i*.001]+[(.1+i*.001)/9]*9) for i in range(200)]
        chosen=participate(rows);self.assertEqual(len(chosen),40)
        self.assertEqual([r['absolute_feature_index'] for r in chosen],list(range(40)))
    def test_old_prototype_mathematics_retained(self):
        self.assertEqual(POLICY['embedding_dim'],128);self.assertEqual(POLICY['target_alignment_weight'],.1)
        self.assertEqual(POLICY['selected_count'],40);self.assertEqual(POLICY['candidate_count'],200)

if __name__=='__main__':unittest.main()
