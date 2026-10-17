# RESERVE BANK OF INDIA: MASTER DIRECTION ON INFORMATION TECHNOLOGY GOVERNANCE AND CYBER SECURITY
**Master Direction: RBI/2023-24/107 DoS.CO.CSITE.SEC.No.3/31.01.015/2023-24**
*Department of Supervision (Cyber Security and Information Technology Examination Cell), RBI, Central Office*

---

## 1. IT GOVERNANCE STRUCTURE

### 1.1 Role of the Board of Directors
- (a) The Board of Directors of the Regulated Entity (RE) holds ultimate responsibility for IT governance, cyber security resilience, and business continuity.
- (b) The Board shall approve an enterprise-wide IT and Cyber Security Policy, reviewing it at least once every financial year.
- (c) The Board shall ensure that adequate budgetary allocations and specialized resources are dedicated to cyber security defense and incident mitigation.

### 1.2 Committees and Organizational Mandate
- (a) **IT Strategy Committee of the Board**: Chaired by an Independent Director with relevant IT/cyber expertise; meets at least quarterly to align IT strategy with business objectives and review major IT procurement and risk profiles.
- (b) **IT Steering Committee**: Executive-level operational committee consisting of business unit heads, Chief Technology Officer (CTO), and Chief Risk Officer (CRO); meets at least monthly to monitor project implementations and infrastructure performance.
- (c) **Information Security Committee (ISC)**: Oversees operational implementation of information security standards, risk assessments, and vulnerability remediation across all branches and digital interfaces.

### 1.3 Chief Information Security Officer (CISO)
- (a) **Appointment and Reporting**: The RE shall designate a senior executive as the Chief Information Security Officer (CISO). To ensure independence and eliminate conflict of interest, the CISO shall NOT report to the Chief Information Officer (CIO) or CTO.
- (b) The CISO shall report directly to the **Chief Executive Officer (MD & CEO)** or the **Audit Committee of the Board (ACB)**.
- (c) **Responsibilities**: The CISO is responsible for formulating cyber security architecture, overseeing Security Operations Centre (SOC) monitoring, managing incident response protocols, and conducting vulnerability testing.

---

## 2. CYBER SECURITY ARCHITECTURE AND DEFENSE

### 2.1 Security Operations Centre (SOC)
- (a) REs shall establish and operate a dedicated 24x7x365 Security Operations Centre (SOC) equipped with automated Security Information and Event Management (SIEM) tools.
- (b) The SOC must continuously monitor network perimeter logs, database access queries, DNS traffic, firewall events, and endpoint telemetry.
- (c) High-severity alert triage must follow documented playbooks, with alert escalation thresholds reviewed semi-annually.

### 2.2 Mandatory Cyber Incident Reporting (The 6-Hour Rule)
- (a) REs shall report all cyber security incidents, breach attempts, ransomware infections, distributed denial of service (DDoS) outages, and unauthorized data exfiltrations to:
  - (i) **Reserve Bank of India (CSITE Cell)**; and
  - (ii) **Indian Computer Emergency Response Team (CERT-In)**.
- (b) **Statutory Reporting Timeline**: The initial notification must be submitted within **six (6) hours** of detection of the incident.
- (c) A comprehensive Root Cause Analysis (RCA) report and impact assessment must be submitted to the RBI within **twenty-one (21) calendar days** of the incident.

### 2.3 Vulnerability Assessment and Penetration Testing (VAPT)
- (a) REs shall conduct comprehensive Vulnerability Assessment and Penetration Testing (VAPT) across all public-facing applications, core banking systems, APIs, and network endpoints:
  - (i) At least once every **six (6) months** for internet-facing systems;
  - (ii) Annually for internal local area networks and database clusters;
  - (iii) Prior to commissioning any new enterprise software, major release, or API integration.
- (b) Penetration testing shall be conducted by independent, CERT-In empanelled cyber security audit organizations.
- (c) Critical and high-risk vulnerabilities must be remediated within thirty (30) days of discovery, with proof of closure validated through re-testing.

### 2.4 Identity and Access Management (IAM)
- (a) Multi-Factor Authentication (MFA) is strictly mandatory for all administrative accounts, database root credentials, remote VPN access, and server console sessions.
- (b) Privileged Access Management (PAM) tools must be implemented to enforce just-in-time credential checkout, session recording, and automatic password rotation every thirty (30) days.
- (c) Principle of Least Privilege (PoLP) and segregation of duties must be enforced across production, testing, and development environments.

---

## 3. THIRD-PARTY AND VENDOR RISK MANAGEMENT

### 3.1 Vendor Due Diligence and Contracts
- (a) Outsourcing of IT functions does NOT diminish the ultimate responsibility of the RE, its Board, and senior management.
- (b) Core banking systems and core decision-making functions regarding credit assessment and risk management shall NOT be outsourced.
- (c) Service Level Agreements (SLAs) with vendors and cloud service providers must explicitly incorporate:
  - (i) **Right to Audit**: Unrestricted access for the RE’s internal auditors, external auditors, and RBI inspection teams to inspect vendor infrastructure, processes, and security logs;
  - (ii) Strict data confidentiality, isolation, and non-disclosure clauses;
  - (iii) Mandatory notification by the vendor of any security breach within **two (2) hours** of occurrence;
  - (iv) Strict adherence to RBI data localization mandates requiring Indian customer data to reside exclusively on servers in India.
- (d) Cloud adoption must adhere to multi-tenant isolation, customer-managed encryption keys (CMEK), and regular disaster recovery (DR) failover drills.
