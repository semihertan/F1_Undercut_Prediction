import fastf1 as ff1
import pandas as pd

from paths import CACHE_DIR, DATA_FILE, ensure_output_dirs


def extract_undercut_rows(years: list[int]) -> pd.DataFrame:
    rows = []

    for year in years:
        print("=" * 48)
        print(f"Processing season: {year}")
        print("=" * 48)

        for round_number in range(1, 26):
            try:
                event = ff1.get_event(year, round_number)
                gp_name = event.EventName
                print(f"{year} round {round_number}: {gp_name}")

                session = ff1.get_session(year, round_number, "R")
                session.load(telemetry=False, weather=False)

                laps = session.laps[
                    [
                        "DriverNumber",
                        "LapNumber",
                        "LapTime",
                        "PitInTime",
                        "Compound",
                        "TyreLife",
                        "TrackStatus",
                        "Time",
                        "Position",
                    ]
                ].copy()

                laps = laps.dropna(subset=["LapTime", "Position", "DriverNumber"])
                laps["Total_Time_Sec"] = laps["Time"].dt.total_seconds()

                invalid_status = ["4", "5", "6", "7"]
                clean_laps = laps[
                    ~laps["TrackStatus"].astype(str).apply(lambda value: any(status in value for status in invalid_status))
                ].copy()
                pit_laps = clean_laps[clean_laps["PitInTime"].notnull()].copy()

                for _, pit in pit_laps.iterrows():
                    driver = pit["DriverNumber"]
                    pit_lap = pit["LapNumber"]
                    previous_lap = pit_lap - 1

                    if previous_lap <= 0:
                        continue

                    previous_lap_data = clean_laps[clean_laps["LapNumber"] == previous_lap]
                    driver_row = previous_lap_data[previous_lap_data["DriverNumber"] == driver]

                    if driver_row.empty:
                        continue

                    driver_position = driver_row["Position"].values[0]
                    if pd.isna(driver_position) or driver_position <= 1.0:
                        continue

                    ahead_row = previous_lap_data[previous_lap_data["Position"] == (driver_position - 1.0)]
                    if ahead_row.empty:
                        continue

                    gap = driver_row["Total_Time_Sec"].values[0] - ahead_row["Total_Time_Sec"].values[0]
                    car_ahead = ahead_row["DriverNumber"].values[0]

                    ahead_pits = pit_laps[
                        (pit_laps["DriverNumber"] == car_ahead) & (pit_laps["LapNumber"] > pit_lap)
                    ]
                    if ahead_pits.empty:
                        continue

                    ahead_pit_lap = ahead_pits.iloc[0]["LapNumber"]
                    if (ahead_pit_lap - pit_lap) > 5:
                        continue

                    compare_lap_data = clean_laps[clean_laps["LapNumber"] == ahead_pit_lap]
                    driver_compare = compare_lap_data[compare_lap_data["DriverNumber"] == driver]
                    ahead_compare = compare_lap_data[compare_lap_data["DriverNumber"] == car_ahead]

                    if driver_compare.empty or ahead_compare.empty:
                        continue

                    success = int(driver_compare["Total_Time_Sec"].values[0] < ahead_compare["Total_Time_Sec"].values[0])

                    rows.append(
                        {
                            "Race": gp_name,
                            "Driver": driver,
                            "Car_Ahead": car_ahead,
                            "Gap": round(gap, 3),
                            "Driver_TyreLife": driver_row["TyreLife"].values[0],
                            "Ahead_TyreLife": ahead_row["TyreLife"].values[0],
                            "Tyre_Advantage": ahead_row["TyreLife"].values[0] - driver_row["TyreLife"].values[0],
                            "Success": success,
                        }
                    )
            except Exception as error:
                print(f"Skipped {year} round {round_number}: {error}")

    return pd.DataFrame(rows)


def main() -> None:
    ensure_output_dirs()
    ff1.Cache.enable_cache(str(CACHE_DIR))

    seasons = [2022, 2023, 2024, 2025]
    dataset = extract_undercut_rows(seasons)
    dataset.to_csv(DATA_FILE, index=False)

    print(f"Dataset rows: {len(dataset)}")
    print(f"Saved: {DATA_FILE}")


if __name__ == "__main__":
    main()
