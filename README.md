# DSC Datathon 2026

## The Mission
We are partnering with Databricks and the United Nations to tackle two critical issues in humanitarian aid. Your work here could directly influence how resources are allocated to millions of people in crisis.

### Challenge 1: The "Geo-Insight" Map
**Question:** *Which Crises Are Most Overlooked?*
Using public funding and severity data, investigate the mismatch between **Humanitarian Needs** (Severity) and **Pooled Fund Coverage** (Funding).
* **Goal:** Identify "Orphan Crises"—regions with high severity but low funding.

### Challenge 2: Smart Beneficiary Targeting
**Question:** *Are these budget ratios realistic?*
Build a tool to flag project proposals with unusually high or low **Cost-Per-Beneficiary** ratios.
* **Goal:** Create a benchmarking tool that clusters similar projects and detects outliers to help vetting teams.

---

## Data Schemas

### 1. CERF Climate Related Allocations
| Column Name | Description |
| :--- | :--- |
| `Year` | The calendar year of the funding decision. |
| `CERF allocations for drought` | Total USD allocated for drought relief. |
| `CERF allocations for floods` | Total USD allocated for flood relief. |
| `CERF allocations for storms` | Total USD allocated for storm relief. |
| `CERF allocations for heat/cold waves` | Total USD allocated for extreme temperature events. |
| `TOTAL climate related allocations` | The sum of all funding for the four climate categories above. |
| `TOTAL CERF allocations (USD)` | The total global CERF funding for the year. |
| `Climate allocation %` | The percentage of the total CERF budget spent specifically on climate-related disasters. |

### 2. CERF Donor Contributions and Allocations - Allocations
| Column Name | Description |
| :--- | :--- |
| `agencyName` | The UN agency receiving the funds. |
| `continentName` | The continent where the operation takes place. |
| `countryCode` | ISO 3-letter country code. |
| `countryName` | The name of the country receiving funds. |
| `dateUSGSignature` | The date the Under-Secretary-General approved the funds. |
| `emergencyTypeName` | The category of the crisis. |
| `projectCode` | The internal reference code for the project. |
| `projectID` | A unique identifier for the specific project entry. |
| `projectTitle` | A descriptive title of the intervention. |
| `regionName` | The specific sub-region. |
| `tableName` | Metadata indicating the source table grouping. |
| `totalAmountApproved` | The total budget allocated in USD. |
| `windowFullName` | The funding stream: Rapid Response (sudden onset) or Underfunded Emergencies (protracted). |
| `year` | The calendar year of the allocation. |
| `projectsectors` | The broad sector of aid. |
| `projectclusters` | The specific coordination group. |
| `projectgroupings` | Additional categorization tags. |
| `projectcapcodes` | Codes linking to the Consolidated Appeals Process. |

### 3. CERF Donor Contributions and Allocations - Contributions
| Column Name | Description |
| :--- | :--- |
| `activityDateType` | Internal code indicating the type of date being tracked. |
| `contributionCode` | The official reference code for the financial transaction. |
| `contributionId` | A unique identifier for the specific contribution entry. |
| `countryCode` | ISO 3-letter code for the donor country. |
| `donor` | The name of the specific donor entity. |
| `donorcommitment` | The amount (USD) the donor has legally committed. |
| `donorpledge` | The amount (USD) promised by the donor. |
| `donorreceived` | The actual cash (USD) transferred and available for use. |
| `donorwriteoff` | Amounts that were pledged or committed but ultimately cancelled or not paid. |
| `latestDate` / `latestDateAsDate` | The date of the most recent update to the transaction status. |
| `flag` | Internal indicator. |
| `donortype` | Classification of the donor. |
| `regionName` | The geopolitical region of the donor. |
| `statusCode` | Current status of the funds. |
| `year` | The fiscal year associated with the contribution. |

### 4. Country Based Pooled Funds (CBPF) - Contributions

| Column Name | Description |
| :--- | :--- |
| `PooledFundId` | Unique ID for the specific Country-Based Pooled Fund. |
| `PooledFundName` | The name of the fund receiving the money. |
| `PooledFundCodeAbbrv` | The short-code for the fund. |
| `ContributionCode` | Unique reference code for the specific financial transaction. |
| `FiscalYear` | The financial year to which the contribution is attributed. |
| `DonorName` | The name of the government or organization providing the funds. |
| `DonorCode` / `GMSDonorID` | Unique identifiers for the donor entity. |
| `CountryCode` | ISO code of the donor country. |
| `PledgeDate` | The date the donor formally promised the funds. |
| `PledgeAmt` | The amount (USD) promised by the donor. |
| `PaidDate` | The date the funds were actually transferred. |
| `PaidAmt` | The actual amount (USD) received. |
| `PipeLineDate` | The date the contribution entered the processing pipeline. |
| `ExpectedDate` | The date the funds were originally expected to arrive. |
| `PledgeAmtLocal` / `PaidAmtLocal` | The pledge/paid amounts in the donor's original currency. |
| `...ExchangeRate` | The exchange rate used to convert the local currency to USD at the time of the transaction. |

### 5. Country Based Pooled Funds (CBPF) - Projects

| Column Name | Description |
| :--- | :--- |
| `PooledFundName` | The specific crisis fund. |
| `AllocationType` | The funding modality. |
| `OrganizationName` / `Type` | The implementing partner. |
| `ProjectTitle` | The official title. |
| `ProjectDuration` | The length of the intervention in months. |
| `Budget` | The total approved budget in USD. |
| `TotalDirectCost` / `SupportCost` | Breakdown of the budget into program delivery vs. administrative overhead. |
| `Men` / `Women` / `Boys` / `Girls` | The target number of direct beneficiaries broken down by demographics. |
| `ProjectSummary` | A narrative description of the project. |
| `IndirectBeneficiaries` | People who benefit broadly but receive no direct service. |
| `CatchmentPopulation` | The total population in the target area, regardless of whether they receive aid. |
| `HumanitarianContext` | Background on the specific crisis situation. |
| `GrantRequestJustification` | The partner's argument for why this project is needed. |
| `Output` / `Activities` | Detailed lists of what will be delivered. |
| `GenderMarker` | A score indicating how well the project addresses gender equality. |
| `ProjectStatus` | Current state of the project. |