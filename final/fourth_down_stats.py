import nfl_data_py as nfl
import pandas as pd

pbp = nfl.import_pbp_data(list(range(2006, 2026)))

# 4th down plays where team went for it (run or pass, not punt/FG)
fourth = pbp[(pbp['down'] == 4) & (pbp['play_type'].isin(['run', 'pass']))]
by_year = fourth.groupby('season').size().reset_index(name='fourth_down_attempts')

# Also get go-for-it rate
all_fourth = pbp[pbp['down'] == 4]
total = all_fourth.groupby('season').size().reset_index(name='total_4th_downs')
merged = by_year.merge(total, on='season')
merged['go_rate_pct'] = (merged['fourth_down_attempts'] / merged['total_4th_downs'] * 100).round(1)

print(merged.to_string(index=False))
