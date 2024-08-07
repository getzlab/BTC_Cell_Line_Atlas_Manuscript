import pandas as pd
import category_encoders as ce
import os

import BTC_Cell_Line_Atlas_Manuscript.scripts.io_library as io_library
from BTC_Cell_Line_Atlas_Manuscript.scripts.io_library import MyLib








# ==================================================================


if __name__ == "__main__":

    print("\nProgram started at", MyLib.get_time())
    print()
# ---------------------------

working_dir = "/Users/nkarisan/PycharmProjects/BTC_Cell_Line_Atlas_Manuscript"
os.chdir(working_dir)

io_library.input_dir = 'input_data/'
io_library.output_dir = ''

# upset_file = 'cell_lines_master_upset_2024-4-13.csv'
# raw_upset_df = MyLib.load_csv(upset_file, index_col=0)

target_genes = ['MDM4 (4194)', 'BRAF (673)', 'ERBB2 (2064)', 'PPCDC (60490)', 'TP63 (8626)', 'FDFT1 (2222)',
                'STAMBP (10617)', 'PRKCI (5584)', 'KIF18B (146909)', 'ITGB5 (3693)', 'CTNNB1 (1499)', 'SREBF1 (6720)',
                'ELMO2 (63916)', 'PRKRA (8575)', 'MAPK1 (5594)', 'MAP2K1 (5604)', 'TGFBR1 (7046)', 'SQLE (6713)',
                'TYMS (7298)', 'DPM2 (8818)', 'SLC7A1 (6541)', 'MBTD1 (54799)', 'FPGS (2356)', 'KRAS (3845)', 'IPPK (64768)',
                'HNF1B (6928)','NXT1 (29107)', 'GRHL2 (79977)', 'UXS1 (80146)', 'ITGAV (3685)', 'TEAD1 (7003)', 'STX4 (6810)',
                'PARD6B (84612)', 'NAMPT (10135)', 'FADD (8772)', 'FERMT2 (10979)', 'EGFR (1956)', 'SOX9 (6662)', 'STXBP3 (6814)',
                'NCKAP1 (10787)', 'SNAP23 (8773)', 'BRK1 (55845)', 'MARK2 (2011)', 'WWTR1 (25937)', 'CRKL (1399)', 'RAB10 (10890)',
                'SLC7A5 (8140)', 'AHCYL1 (10768)', 'SLC2A1 (6513)', 'RFK (55312)', 'ARHGEF7 (8874)', 'SLC3A2 (6520)', 'ITGB1 (3688)',
                'RIC1 (57589)', 'TUBB4B (10383)', 'CRK (1398)', 'RAB6A (5870)', 'HDAC7 (51564)', 'ITGA3 (3675)',
                'CDK4 (1019)', 'IDH3A (3419)', 'DHODH (1723)', 'NMNAT1 (64802)', 'ABI1 (10006)', 'CDH1 (999)', 'PIK3CA (5290)',
                'PTPN11 (5781)', 'PRMT5 (10419)'
                ]
target_genes = sorted(target_genes)
# ---------------------------
print("\nProgram finished at", MyLib.get_time())