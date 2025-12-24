from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import re


class ClimateDataLoader:

    VARIABLE_ALIASES = {
        "t2m": ["t2m", "2t", "temperature", "temp"],
        "tp": ["tp", "totalprecipitation", "precipitation", "rain", "rainfall"]
    }

    CSV_TIME_ALIASES = ["time", "date", "datetime", "timestamp"]

    # --------------------------------------------------
    def _normalize(self, name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    # --------------------------------------------------
    def _resolve_variable(self, ds, requested):
        if requested in ds.variables:
            return requested
        for alias in self.VARIABLE_ALIASES.get(requested, []):
            if alias in ds.variables:
                return alias
        raise KeyError(
            f"Variable '{requested}' not found.\n"
            f"Available variables: {list(ds.variables.keys())}"
        )

    # ==================================================
    # ERA5 TIME SERIES (CENTER GRID CELL)
    # ==================================================
    def load_variable(self, file_path, variable):
        ds = Dataset(file_path)

        var_name = self._resolve_variable(ds, variable)
        var = ds.variables[var_name]

        lat = var.shape[-2] // 2
        lon = var.shape[-1] // 2

        data = np.asarray(var[:, lat, lon], dtype=float)

        time_var = None
        for key in ("time", "valid_time"):
            if key in ds.variables:
                time_var = ds.variables[key]
                break

        if time_var is None:
            ds.close()
            raise KeyError("No valid time variable found in NetCDF file.")

        times = num2date(
            time_var[:],
            units=time_var.units,
            calendar=getattr(time_var, "calendar", "standard"),
            only_use_cftime_datetimes=True
        )

        ds.close()

        # Kelvin → Celsius
        if variable == "t2m" and np.nanmean(data) > 100:
            data = data - 273.15

        return data, np.asarray(times)

    # ==================================================
    # ERA5 SPATIAL FIELD (LAT × LON)
    # ==================================================
    def load_variable_2d(self, file_path, variable, time_index=0):
        ds = Dataset(file_path)

        var_name = self._resolve_variable(ds, variable)
        var = ds.variables[var_name]

        field = np.asarray(var[time_index, :, :], dtype=float)
        lats = ds.variables["latitude"][:]
        lons = ds.variables["longitude"][:]

        if variable == "t2m" and np.nanmean(field) > 100:
            field = field - 273.15

        ds.close()
        return lons, lats, field

    # ==================================================
    # CSV LOADER (FIXED)
    # ==================================================
    def load_csv(self, file_path, column_name):
        df = pd.read_csv(file_path)

        if column_name not in df.columns:
            raise KeyError(
                f"Column '{column_name}' not found in CSV.\n"
                f"Available columns: {list(df.columns)}"
            )

        # ---- FORCE FLOAT (CRITICAL FIX) ----
        data = pd.to_numeric(df[column_name], errors="coerce").astype(float).values

        # ---- TIME COLUMN AUTO-DETECT ----
        norm_cols = {self._normalize(c): c for c in df.columns}
        time_col = None

        for alias in self.CSV_TIME_ALIASES:
            if alias in norm_cols:
                time_col = norm_cols[alias]
                break

        if time_col:
            times = pd.to_datetime(df[time_col], errors="coerce")
        else:
            times = pd.date_range(
                start="2000-01-01",
                periods=len(df),
                freq="D"
            )

        # Kelvin → Celsius safety
        if np.nanmean(data) > 100:
            data = data - 273.15

        return data, times.to_numpy()
