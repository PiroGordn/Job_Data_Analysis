import pandas as pd
import os


def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "dataset.xlsx")

    print("Loading from:", file_path)
    print("Reading file...")

    df = pd.read_excel(file_path, engine='openpyxl')

    print("File loaded successfully")

    df.columns = df.columns.str.strip()

    print("Processing dates...")
    df['Job Posting Date'] = pd.to_datetime(df['Job Posting Date'], errors='coerce')

    print("Processing salary...")
    salary = df['Salary Range'].str.extract(r'(\d+)-(\d+)')
    df['min_salary'] = pd.to_numeric(salary[0], errors='coerce')
    df['max_salary'] = pd.to_numeric(salary[1], errors='coerce')
    df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

    print("Processing skills...")
    df['skills_list'] = df['skills'].fillna('').str.lower().str.split(',')

    print("Data processing done")

    return df


def get_top_skills(df):
    all_skills = []
    for skills in df['skills_list']:
        all_skills.extend(skills)

    skill_counts = pd.Series(all_skills).value_counts().head(10)
    return skill_counts.to_dict()

def jobs_over_time(df):
    df = df.copy()
    df = df.dropna(subset=['Job Posting Date'])

    if df.empty:
        return {}

    trend = df.groupby(df['Job Posting Date'].dt.to_period('M')).size()

    start = df['Job Posting Date'].min()
    end = df['Job Posting Date'].max()

    if pd.isna(start) or pd.isna(end):
        return {}

    full_range = pd.period_range(
        start=start.to_period('M'),
        end=end.to_period('M'),
        freq='M'
    )

    trend = trend.reindex(full_range, fill_value=0)

    return {str(k): int(v) for k, v in trend.items()}


def salary_by_role(df):
    return df.groupby('Job Title')['avg_salary'].mean().dropna().head(10).to_dict()


def job_distribution(df, max_roles=20, target_other_ratio=0.15):
    counts = df['Job Title'].dropna().value_counts()
    if counts.empty:
        return {'labels': [], 'values': []}

    total = int(counts.sum())
    top_counts = counts.head(max_roles)
    kept_total = int(top_counts.sum())
    current_roles = len(top_counts)

    while current_roles < len(counts):
        other_count = total - kept_total
        other_ratio = (other_count / total) if total else 0
        if other_ratio <= target_other_ratio:
            break
        current_roles += 1
        kept_total += int(counts.iloc[current_roles - 1])

    final_counts = counts.head(current_roles)
    labels = list(final_counts.index)
    values = [int(v) for v in final_counts.values]

    other_count = total - kept_total
    if other_count > 0:
        labels.append('Other')
        values.append(other_count)

    return {'labels': labels, 'values': values}


def salary_distribution_box(df, top_n=6):
    role_counts = df['Job Title'].dropna().value_counts().head(top_n)
    result = []

    for role in role_counts.index:
        salaries = df.loc[df['Job Title'] == role, 'avg_salary'].dropna().tolist()
        if salaries:
            result.append({'role': role, 'salaries': salaries})

    return result


def salary_by_location(df, top_n=10):
    data = (
        df.groupby('location')['avg_salary']
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return {str(k): float(v) for k, v in data.items()}


def qualification_heatmap(df, top_titles=10):
    qualification_col = None
    for col in df.columns:
        if str(col).strip().lower() in {'qualifications', 'qualification', 'education', 'degree'}:
            qualification_col = col
            break

    if qualification_col is None or 'Job Title' not in df.columns:
        return {'x': [], 'y': [], 'z': []}

    working = df[['Job Title', qualification_col]].dropna()
    if working.empty:
        return {'x': [], 'y': [], 'z': []}

    top_job_titles = working['Job Title'].value_counts().head(top_titles).index.tolist()
    working = working[working['Job Title'].isin(top_job_titles)].copy()

    working[qualification_col] = working[qualification_col].astype(str).str.strip()

    top_qualifications = (
        working[qualification_col]
        .value_counts()
        .head(8)
        .index
        .tolist()
    )

    working = working[working[qualification_col].isin(top_qualifications)]
    if working.empty:
        return {'x': [], 'y': [], 'z': []}

    pivot = pd.crosstab(working['Job Title'], working[qualification_col])
    pivot = pivot.reindex(index=top_job_titles, fill_value=0)
    pivot = pivot.reindex(columns=top_qualifications, fill_value=0)

    return {
        'x': list(pivot.columns),
        'y': list(pivot.index),
        'z': pivot.values.tolist()
    }


def job_distribution_by_work_type(df, top_n=6):
    col_name = None
    for col in df.columns:
        if str(col).strip().lower() in {'work type', 'worktype', 'employment type'}:
            col_name = col
            break

    if col_name is None:
        return {'labels': [], 'values': []}

    counts = (
        df[col_name]
        .dropna()
        .astype(str)
        .str.strip()
        .replace('', pd.NA)
        .dropna()
        .value_counts()
    )

    if counts.empty:
        return {'labels': [], 'values': []}

    top_counts = counts.head(top_n)
    other_count = int(counts.iloc[top_n:].sum()) if len(counts) > top_n else 0

    labels = list(top_counts.index)
    values = [int(v) for v in top_counts.values]

    if other_count > 0:
        labels.append('Other')
        values.append(other_count)

    return {'labels': labels, 'values': values}