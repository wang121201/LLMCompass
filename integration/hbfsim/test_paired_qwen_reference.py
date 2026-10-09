"""CPU-only accounting guards for the paired Qwen hardware driver."""
import unittest
import paired_qwen_reference as driver


class Guards(unittest.TestCase):
    def table(self, rows):
        header = '"ID","Metric Name","Metric Unit","Metric Value"\n'
        return header + '\n'.join(','.join('"' + str(x) + '"' for x in row) for row in rows)

    def test_counter_parser(self):
        table = self.table([(0, driver.METRICS[0], 'byte', '1,024'),
                            (0, driver.METRICS[1], 'byte', '256')])
        self.assertEqual(driver.parse_counters(table, 1)[0]['read_bytes'], 1024)

    def test_missing_range(self):
        with self.assertRaises(RuntimeError):
            driver.parse_counters(self.table([(0, driver.METRICS[0], 'byte', 1),
                (0, driver.METRICS[1], 'byte', 2)]), 5)

    def test_duplicate_metric(self):
        with self.assertRaises(RuntimeError):
            driver.parse_counters(self.table([(0, driver.METRICS[0], 'byte', 1),
                (0, driver.METRICS[0], 'byte', 2)]), 1)

    def test_wrong_unit(self):
        with self.assertRaises(RuntimeError):
            driver.parse_counters(self.table([(0, driver.METRICS[0], 'Kbyte', 1)]), 1)

    def test_wide_table(self):
        table = ('"ID","dram__bytes_read.sum","dram__bytes_write.sum"\n'
                 '"","byte","byte"\n"0","9,242,692,224","73,309,440"\n')
        self.assertEqual(driver.parse_counters(table, 1)[0]['write_bytes'], 73309440)

    def test_wide_wrong_unit(self):
        with self.assertRaises(RuntimeError):
            driver.parse_counters('"ID","dram__bytes_read.sum","dram__bytes_write.sum"\n'
                                  '"","Kbyte","byte"\n"0","1","2"', 1)

    def test_matrix(self):
        self.assertEqual(len(set(driver.CASES)), 8)
        self.assertEqual(sum(d + 1 for p, d in driver.CASES), 46)

    def test_denominator(self):
        values = driver.statistics_ms([1.0, 2.0, 3.0])
        self.assertEqual(values['median'], 2.0)
        with self.assertRaises(RuntimeError): driver.statistics_ms([0])


if __name__ == '__main__':
    unittest.main()
