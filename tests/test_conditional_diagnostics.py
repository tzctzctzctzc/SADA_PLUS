import unittest

from libs.trainer.conditional_diagnostics import (
    build_conditional_class_report,
)


class ConditionalClassDiagnosticTests(unittest.TestCase):
    def test_report_isolates_vlm_rescue_without_source_support(self):
        prefix = 'conditional_class_l2_c7_'
        report = build_conditional_class_report(
            {
                prefix + 'source_samples': 0,
                prefix + 'target_samples': 5,
                prefix + 'vlm_rescue_samples': 3,
                prefix + 'vlm_rescue_candidate_samples': 4,
                prefix + 'vlm_forgetting_samples': 2,
                prefix + 'vlm_forgetting_candidate_samples': 3,
                prefix + 'source_present': 0,
                prefix + 'target_present': 2,
                prefix + 'target_loss': 1.6,
                prefix + 'target_objective': 0.4,
                prefix + 'target_samples_without_source': 5,
                prefix + 'vlm_rescue_samples_without_source': 3,
                prefix + 'target_objective_without_source': 0.4,
                prefix + 'vlm_residual_objective': 0.02,
                prefix + 'vlm_residual_budget_excess': 0,
                prefix + 'vlm_forgetting_objective': 0.04,
                prefix + 'vlm_forgetting_budget_excess': 0,
            },
            batch_count=4,
            epoch=3,
        )

        self.assertEqual(report['schema_version'], 'conditional-class-support-diagnostic-v3')
        self.assertEqual(report['epoch'], 3)
        row = report['rows'][0]
        self.assertEqual(row['level'], 2)
        self.assertEqual(row['class_id'], 7)
        self.assertEqual(row['target_samples_without_source_total'], 5)
        self.assertEqual(row['vlm_rescue_samples_without_source_total'], 3)
        self.assertEqual(row['vlm_rescue_candidate_samples_total'], 4)
        self.assertEqual(row['vlm_forgetting_samples_total'], 2)
        self.assertEqual(row['vlm_forgetting_candidate_samples_total'], 3)
        self.assertAlmostEqual(row['target_loss_mean_when_present'], 0.8)
        self.assertAlmostEqual(row['target_objective_mean_per_batch'], 0.1)
        self.assertAlmostEqual(
            row['vlm_forgetting_objective_mean_per_batch'], 0.01
        )
        self.assertEqual(report['summary']['target_without_source_rate'], 1.0)
        self.assertEqual(report['summary']['vlm_rescue_without_source_rate'], 1.0)
        self.assertEqual(report['summary']['vlm_forgetting_samples_total'], 2)
        self.assertEqual(
            report['summary']['vlm_forgetting_candidate_samples_total'], 3
        )
        self.assertEqual(
            report['summary']['target_objective_without_source_fraction'], 1.0
        )
        self.assertAlmostEqual(
            report['summary']['vlm_residual_objective_mean_per_batch'], 0.005
        )
        self.assertEqual(
            report['summary']['vlm_residual_budget_excess_total'], 0.0
        )
        self.assertAlmostEqual(
            report['summary']['vlm_forgetting_objective_mean_per_batch'], 0.01
        )
        self.assertEqual(
            report['summary']['vlm_forgetting_budget_excess_total'], 0.0
        )

    def test_report_keeps_supported_classes_separate(self):
        report = build_conditional_class_report(
            {
                'conditional_class_l0_c1_source_samples': 4,
                'conditional_class_l0_c1_target_samples': 2,
                'conditional_class_l0_c1_vlm_rescue_samples': 1,
                'conditional_class_l0_c1_source_present': 1,
                'conditional_class_l0_c1_target_present': 1,
                'conditional_class_l0_c1_target_loss': 0.5,
                'conditional_class_l0_c1_target_objective': 0.25,
                'conditional_class_l0_c1_target_samples_without_source': 0,
                'conditional_class_l0_c1_vlm_rescue_samples_without_source': 0,
                'conditional_class_l0_c1_target_objective_without_source': 0,
            },
            batch_count=1,
            epoch=1,
        )

        self.assertEqual(report['summary']['target_without_source_rate'], 0.0)
        self.assertEqual(report['summary']['vlm_rescue_without_source_rate'], 0.0)
        self.assertEqual(
            report['summary']['target_objective_without_source_fraction'], 0.0
        )


if __name__ == '__main__':
    unittest.main()
