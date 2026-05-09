import fastf1

from paths import CACHE_DIR, SAMPLE_FILE, ensure_output_dirs


def main() -> None:
    ensure_output_dirs()
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    print("Preparing sample race snapshot: 2023 Dutch Grand Prix")
    session = fastf1.get_session(2023, "Zandvoort", "R")
    session.load(telemetry=False, weather=False)

    columns = ["Driver", "LapNumber", "Stint", "Compound", "TyreLife", "Position", "LapTime", "PitOutTime"]
    sample = session.laps[session.laps["Driver"].isin(["VER", "ALO"])].head(15).copy()
    sample = sample[columns]
    sample["Is_Pit_Stop"] = sample["PitOutTime"].notnull()
    sample["LapTime"] = sample["LapTime"].dt.total_seconds().round(3)

    sample.to_csv(SAMPLE_FILE, index=False)

    print(sample.to_string(index=False))
    print(f"Saved: {SAMPLE_FILE}")


if __name__ == "__main__":
    main()
