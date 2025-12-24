import numpy as np
import pandas as pd


class MonthlyAggregation:
    """
    Monthly mean aggregation (Jan–Dec).
    Works with:
    - datetime objects
    - pandas datetime
    - numeric index (fallback)
    """

    def aggregate(self, data, times):
        data = np.asarray(data)

        # --- Try datetime-based monthly aggregation ---
        try:
            times = pd.to_datetime(times)
            months = times.month
        except Exception:
            # --- Fallback: numeric index → fake months ---
            months = (np.arange(len(data)) % 12) + 1

        monthly_means = []
        for m in range(1, 13):
            mask = months == m
            if np.any(mask):
                monthly_means.append(np.nanmean(data[mask]))
            else:
                monthly_means.append(np.nan)

        x = np.arange(1, 13)
        return x, np.array(monthly_means)


class HourlyAggregation:
    """
    No aggregation.
    Returns raw time index and values.
    """

    def aggregate(self, data, times):
        data = np.asarray(data)
        x = np.arange(len(data))
        return x, data
