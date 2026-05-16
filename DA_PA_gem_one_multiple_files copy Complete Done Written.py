import pandas as pd
import os
import itertools
import gc
import psutil 

# ==========================================
# CLASS 1: DataProfile (Micro-Modular Data Model)
# ==========================================
class DataProfile:
    """Loads a dataset, extracts its profile using isolated, fault-tolerant methods."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.is_valid = False
        self.error_message = ""
        
        # Profile Data Storage
        self.num_rows = 0
        self.num_columns = 0
        self.columns = []
        self.metadata = {}        
        self.top_frequent = {}    
        self.unique_value_sets = {} 
        
        self._orchestrate_profiling()

    def _orchestrate_profiling(self):
        """The internal manager. It calls sub-functions step-by-step."""
        df = self._load_dataframe()
        if df is None:
            return # Stop here if the file wouldn't even load

        self.num_rows = len(df)
        self.num_columns = len(df.columns)
        self.columns = df.columns.tolist()

        # Isolate column processing so one bad column doesn't break the file
        for col in self.columns:
            self._extract_metadata(df, col)
            self._extract_top_frequent(df, col)
            self._extract_unique_sets(df, col)

        self.is_valid = True
        
        # Free RAM
        del df
        gc.collect()

    # --- Micro-Modules ---
    def _load_dataframe(self):
        """Sole responsibility: Load the file into Pandas."""
        file_lower = self.file_path.lower()
        try:
            if file_lower.endswith(('.csv', '.csv.gz')): return pd.read_csv(self.file_path)
            elif file_lower.endswith('.json'): return pd.read_json(self.file_path)
            elif file_lower.endswith(('.xls', '.xlsx')): return pd.read_excel(self.file_path)
            elif file_lower.endswith('.parquet'): return pd.read_parquet(self.file_path)
            else:
                self.error_message = "Unsupported format."
                return None
        except Exception as e:
            self.error_message = f"File load failed: {e}"
            return None

    def _extract_metadata(self, df, col):
        """Sole responsibility: Calculate basic column stats."""
        try:
            unique_count = df[col].nunique(dropna=False)
            is_pk = (unique_count == self.num_rows) and (df[col].isna().sum() == 0)

            self.metadata[col] = {
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "unique": unique_count,
                "is_primary_key": is_pk
            }
        except Exception as e:
            self.metadata[col] = {"dtype": "ERROR", "missing": 0, "unique": 0, "is_primary_key": False}

    def _extract_top_frequent(self, df, col):
        """Sole responsibility: Find the top 5 values."""
        try:
            counts = df[col].value_counts(dropna=False)
            top_counts = counts.head(5)
            
            top_list = [{"value": v, "count": c, "pct": round((c / self.num_rows) * 100, 2)} 
                        for v, c in top_counts.items()]
            others_pct = ((self.num_rows - top_counts.sum()) / self.num_rows) * 100
            
            self.top_frequent[col] = {"top_n": top_list, "others_pct": round(others_pct, 2)}
        except Exception as e:
            self.top_frequent[col] = {"top_n": [], "others_pct": 0.0}

    def _extract_unique_sets(self, df, col):
        """Sole responsibility: Extract lightweight sets for relationship mapping."""
        try:
            self.unique_value_sets[col] = set(df[col].dropna().astype(str).str.strip().str.lower())
        except Exception as e:
            self.unique_value_sets[col] = set()


# ==========================================
# CLASS 2: RelationshipEngine (The Logic Layer)
# ==========================================
class RelationshipEngine:
    """Calculates architectural relationships between DataProfiles."""
    def __init__(self, profiles: list):
        self.profiles = {p.filename: p for p in profiles if p.is_valid}
        self.relationships = []

    def analyze(self):
        if len(self.profiles) < 2: return self.relationships

        for name_a, name_b in itertools.combinations(self.profiles.keys(), 2):
            prof_a = self.profiles[name_a]
            prof_b = self.profiles[name_b]
            
            cols_a = set(prof_a.columns)
            cols_b = set(prof_b.columns)
            
            shared_cols = list(cols_a.intersection(cols_b))
            exclusive_a = list(cols_a - cols_b)
            exclusive_b = list(cols_b - cols_a)
            
            if not shared_cols: continue

            col_overlaps = {}
            zero_overlaps = {}
            
            for col in shared_cols:
                vals_a = prof_a.unique_value_sets.get(col, set())
                vals_b = prof_b.unique_value_sets.get(col, set())
                
                if not vals_a or not vals_b: continue 
                    
                shared_vals = vals_a.intersection(vals_b)
                overlap_count = len(shared_vals)
                
                if overlap_count > 0:
                    pct_a = (overlap_count / len(vals_a)) * 100
                    pct_b = (overlap_count / len(vals_b)) * 100
                    
                    if prof_a.metadata[col]['is_primary_key'] and pct_b >= 95:
                        rel_type = f"Foreign Key (Child {name_b} -> Parent {name_a})"
                    elif prof_b.metadata[col]['is_primary_key'] and pct_a >= 95:
                        rel_type = f"Foreign Key (Child {name_a} -> Parent {name_b})"
                    elif pct_a >= 95 and pct_b >= 95:
                        rel_type = "Strong Two-Way Link"
                    elif pct_a < 20 and pct_b < 20:
                        rel_type = "Incidental Overlap"
                    else:
                        rel_type = "Partial Overlap"
                        
                    col_overlaps[col] = {
                        "shared_count": overlap_count,
                        "total_a": len(vals_a), "total_b": len(vals_b),
                        "pct_a": round(pct_a, 1), "pct_b": round(pct_b, 1),
                        "type": rel_type, "score": max(pct_a, pct_b)
                    }
                else:
                    zero_overlaps[col] = {"in_a": len(vals_a), "in_b": len(vals_b)}
            
            if col_overlaps or zero_overlaps:
                ratio = f"{round(prof_a.num_rows/prof_b.num_rows, 1)}:1" if prof_a.num_rows >= prof_b.num_rows else f"1:{round(prof_b.num_rows/prof_a.num_rows, 1)}"
                
                self.relationships.append({
                    "file_a": name_a, "file_b": name_b,
                    "rows_a": prof_a.num_rows, "rows_b": prof_b.num_rows,
                    "ratio": ratio,
                    "shared_cols": shared_cols,
                    "exc_a": exclusive_a, "exc_b": exclusive_b,
                    "overlaps": col_overlaps, "zeros": zero_overlaps
                })
                
        return self.relationships


# ==========================================
# CLASS 3: ReportFormatter (The UI Layer)
# ==========================================
class ReportFormatter:
    """Strictly handles text formatting and console printing."""
    
    @staticmethod
    def print_profile(profile: DataProfile):
        print("\n" + "="*85)
        print(f" PROFILE FOR: {profile.filename.upper()}")
        print("="*85)
        print(f"Rows: {profile.num_rows:,} | Columns: {profile.num_columns}")
        
        print(f"\n[METADATA]")
        print(f"{'Column Name':<25} | {'Type':<10} | {'NaNs':<10} | {'Uniques':<10} | {'Primary Key'}")
        print("-" * 85)
        for col in profile.columns:
            meta = profile.metadata.get(col, {})
            if not meta: continue
            pk_flag = "⭐ YES" if meta.get('is_primary_key') else "No"
            print(f"{col:<25} | {meta.get('dtype', 'N/A'):<10} | {meta.get('missing', 0):<10,} | {meta.get('unique', 0):<10,} | {pk_flag}")

        print(f"\n[TOP 5 FREQUENT VALUES]")
        for col, data in profile.top_frequent.items():
            if not data.get('top_n'): continue
            print(f"\n-> Column: {col}")
            print(f"   {'Value':<25} | {'Count':<10} | {'Percentage'}")
            print("   " + "-" * 55)
            
            for stat in data['top_n']:
                val_str = str(stat['value'])
                if len(val_str) > 22: val_str = val_str[:19] + '...'
                print(f"   {val_str:<25} | {stat['count']:<10,} | {stat['pct']}%")
                
            if data['others_pct'] > 0:
                print("   " + "-" * 55)
                print(f"   {'[All Other Values]':<25} | {'-':<10} | {data['others_pct']}%")

    @staticmethod
    def print_relationships(relationships: list):
        print("\n" + "="*90)
        print(" CROSS-DATASET INTELLIGENCE REPORT")
        print("="*90)
        
        if not relationships:
            print("\n   -> No significant relationships found.")
            return

        for rel in relationships:
            print(f"\n[🔗] {rel['file_a']}  <--->  {rel['file_b']}")
            print(f"     Row Ratio: {rel['ratio']} | Total Rows: {rel['rows_a']:,} vs {rel['rows_b']:,}")
            print(f"     Shared Columns ({len(rel['shared_cols'])}): {', '.join(rel['shared_cols'])}")
            
            if rel['exc_a']:
                print(f"     Exclusive to {rel['file_a']}: {', '.join(rel['exc_a'][:6])}" + ("..." if len(rel['exc_a']) > 6 else ""))
            if rel['exc_b']:
                print(f"     Exclusive to {rel['file_b']}: {', '.join(rel['exc_b'][:6])}" + ("..." if len(rel['exc_b']) > 6 else ""))
            
            if rel['overlaps']:
                print("     " + "-" * 80)
                sorted_overlaps = sorted(rel['overlaps'].items(), key=lambda x: x[1]['score'], reverse=True)
                for col, stats in sorted_overlaps:
                    print(f"     ✅ {col} [{stats['type']}]")
                    print(f"        {stats['pct_a']}% ({stats['shared_count']:,} of {stats['total_a']:,}) of {rel['file_a']} is in {rel['file_b']}")
                    print(f"        {stats['pct_b']}% ({stats['shared_count']:,} of {stats['total_b']:,}) of {rel['file_b']} is in {rel['file_a']}")
            
            if rel['zeros']:
                print("     " + "-" * 80)
                for col, stats in rel['zeros'].items():
                    print(f"     ❌ {col} [MUTUALLY EXCLUSIVE]")
                    print(f"        Contains {stats['in_a']:,} values in A, {stats['in_b']:,} values in B. NO OVERLAP.")


# ==========================================
# CLASS 4: AppOrchestrator (The Main Controller)
# ==========================================
class AppOrchestrator:
    """Manages file discovery, dynamic hardware checks, and coordinates classes."""
    def __init__(self, directory: str):
        self.directory = directory
        self.supported = ['.csv', '.csv.gz', '.json', '.xls', '.xlsx', '.parquet']
        self.pandas_ram_multiplier = 4.0 

    def _perform_pre_flight_check(self, valid_files: list) -> bool:
        print("\n" + "="*85)
        print(" 🖥️  SYSTEM PRE-FLIGHT DIAGNOSTICS")
        print("="*85)
        
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024**3)
        available_ram_gb = vm.available / (1024**3)
        
        print(f"[HARDWARE PROFILE]")
        print(f"Total System RAM : {total_ram_gb:.1f} GB")
        print(f"Available RAM    : {available_ram_gb:.1f} GB (Currently Free)")
        print("-" * 85)
        
        print(f"[DATA PAYLOAD]")
        total_size_bytes = 0
        
        for filename in valid_files:
            file_path = os.path.join(self.directory, filename)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size_bytes += os.path.getsize(file_path)
            print(f"  -> {filename:<30} | Size: {file_size_mb:>8.2f} MB")
            
        total_size_mb = total_size_bytes / (1024 * 1024)
        total_size_gb = total_size_bytes / (1024**3)
        expected_ram_gb = total_size_gb * self.pandas_ram_multiplier
        
        print("-" * 85)
        print(f"Total Disk Size  : {total_size_mb:,.2f} MB ({total_size_gb:.2f} GB)")
        print(f"Expected RAM Load: ~{expected_ram_gb:.2f} GB (Based on 4x Pandas multiplier)")
        print("="*85)
        
        safe_capacity_gb = available_ram_gb - 1.0 
        
        if expected_ram_gb > safe_capacity_gb:
            print("\n" + "!"*85)
            print(" 🚨 CRITICAL RESOURCE WARNING: CAPACITY EXCEEDED 🚨")
            print("!"*85)
            print(f"   -> Your system only has {available_ram_gb:.1f} GB of RAM available.")
            print(f"   -> Processing this batch requires ~{expected_ram_gb:.2f} GB of RAM.")
            override = input("\nDo you want to override this warning and try anyway? (y/N): ")
            if override.lower() != 'y':
                print("[ABORTED] Batch processing canceled by safety protocol.")
                return False
        else:
            print(f"\n✅ Diagnostics Passed: System has sufficient RAM to process this batch.")
            
        return True

    def run(self):
        if not os.path.exists(self.directory):
            print(f"[ERROR] Directory not found: {self.directory}")
            return

        valid_files = [f for f in os.listdir(self.directory) if any(f.lower().endswith(e) for e in self.supported)]
        
        if not valid_files:
            print(f"No supported files found in '{self.directory}'.")
            return
            
        if not self._perform_pre_flight_check(valid_files):
            return 

        print("\nInitiating Data Profiling Sequence...")
        profiles = []
        for i, filename in enumerate(valid_files, 1):
            print(f"[{i}/{len(valid_files)}] Profiling {filename}...")
            
            profile = DataProfile(os.path.join(self.directory, filename))
            
            if profile.is_valid:
                ReportFormatter.print_profile(profile)
                profiles.append(profile)
            else:
                print(f"   -> [FAILED] {profile.error_message}")

        if len(profiles) > 1:
            print("\nBuilding Cross-Dataset Relationships...")
            engine = RelationshipEngine(profiles)
            relationships = engine.analyze()
            ReportFormatter.print_relationships(relationships)


if __name__ == "__main__":
    app = AppOrchestrator("./data_folder")
    app.run()