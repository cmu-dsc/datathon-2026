## Challenge Prompt

This folder contains the data relevant to the Geo-Insight Challenge.

Integrate crisis severity indices with global funding data to determine mismatch between humanitarian need and actual resource allocation, helping to identify which global crises remain critically overlooked by current funding streams.

---

## Dataset Links

Humanitarian Response Plans: https://data.humdata.org/dataset/humanitarian-response-plans
OCHA Global Subnational Population Statistics: https://data.humdata.org/dataset/cod-ps-global
Global Humanitarian Programme Cycle, Humanitarian Needs: https://data.humdata.org/dataset/global-hpc-hno

---

## Data Schemas

### 1. Humanitarian Response Plans

| Column Name | Description |
| :--- | :--- |
| `code` | The official identifier for the Humanitarian Response Plan. |
| `internalId` | A unique internal ID used for system tracking. |
| `startDate` | The date the response plan officially begins. |
| `endDate` | The date the response plan is scheduled to conclude. |
| `planVersion` | The official name or title of the response plan. |
| `categories` | The classification of the emergency or response type. |
| `locations` | ISO country codes for the areas covered by the plan. |
| `years` | The specific calendar years the plan is active. |
| `origRequirements` | The total initial funding requested in USD. |
| `revisedRequirements` | The updated total funding requested in USD. |

### 2. OCHA Global Subnational Population Statistics

| Column Name | Description |
| :--- | :--- |
| `ISO3` | The 3-letter ISO country code. |
| `Country` | The full name of the country. |
| `ADM1_PCODE` / `ADM1_NAME` | The unique P-code and name for the first administrative level (e.g., Province or State). |
| `ADM2_PCODE` / `ADM2_NAME` | The unique P-code and name for the second administrative level (e.g., District). |
| `ADM3_PCODE` / `ADM3_NAME` | The unique P-code and name for the third administrative level. |
| `ADM4_PCODE` / `ADM4_NAME` | The unique P-code and name for the fourth administrative level. |
| `Population_group` | The specific population segment being counted (e.g., "Total," "IDP," "Refugee"). |
| `Gender` | The gender category associated with the count (e.g., "f", "m", "all"). |
| `Age_range` | The specific age bracket for the count (e.g., "0-4", "18-59"). |
| `Age_min` / `Age_max` | The numeric lower and upper bounds of the age range. |
| `Population` | The total number of people in the specified demographic group and location. |
| `Reference_year` | The year the data was collected or projected for. |
| `Source` | The organization that provided the original data. |
| `Contributor` | The entity responsible for uploading or managing the dataset on HDX. |

### 3. Global Humanitarian Programme Cycle, Humanitarian Needs

| Column Name | Description |
| :--- | :--- |
| `Country ISO3` | The 3-letter ISO country code. |
| `Admin 1 PCode` / `Admin 1 Name` | Unique identifier and name for the first administrative level (e.g., Province). |
| `Admin 2 PCode` / `Admin 2 Name` | Unique identifier and name for the second administrative level (e.g., District). |
| `Admin 3 PCode` / `Admin 3 Name` | Unique identifier and name for the third administrative level. |
| `Description` | The text description of the specific humanitarian sector (e.g., "Emergency Shelter"). |
| `Cluster` | The specific code for the coordination cluster (e.g., "SHL" for Shelter). |
| `Category` | The classification category for the data entry. |
| `Population` | The total baseline population in the specified area. |
| `In Need` | The number of people assessed to be in need of humanitarian assistance. |
| `Targeted` | The number of people the response plan aims to assist. |
| `Affected` | The number of people affected by the crisis (a subset of the total population). |
| `Reached` | The number of people who have actually received assistance (if available). |
| `Info` | Additional metadata or informational notes. |