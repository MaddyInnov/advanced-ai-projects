"""Generates the 14-document corpus. Run once; corpus files are committed artefacts."""
import pathlib
OUT = pathlib.Path(__file__).parent / "corpus"
OUT.mkdir(exist_ok=True)

DOCS = {}

DOCS["password-policy.md"] = """# Password Policy

## Purpose
This Password Policy defines how passwords are created, changed, stored, and protected at the company. Strong passwords are the first line of defence against credential theft, brute force attacks, and account takeover. This policy applies to all employees, contractors, and service accounts that access company systems.

## Minimum Length and Complexity
All user passwords must be at least 14 characters long. Passwords must contain a mix of uppercase letters, lowercase letters, numbers, and special symbols. Passphrases of four or more unrelated words are encouraged because they are longer and easier to remember. Passwords must not contain the user's name, the company name, a dictionary word used alone, or a previously breached password.

## Password Expiry and History
User passwords expire every 90 days and must be changed before expiry. The system remembers the last 10 passwords, and a new password must not match any of those previous 10 passwords. Service account passwords expire every 180 days and are managed by the IT security team in the central vault.

## Lockout and Failed Logins
After 5 consecutive failed login attempts, an account is locked for 30 minutes. Helpdesk staff can unlock an account only after verifying the caller's identity with two independent identity checks. Repeated lockouts from the same source address must be reported to the security operations centre as a possible brute force attack.

## Storage and Sharing
Passwords must never be written on paper, stored in plain text files, sent by email, or shared with another person, including a manager. Employees must use the approved enterprise password manager to store work credentials. Passwords for shared administrative systems are stored only in the privileged access vault and are checked out for a single session at a time.

## Responsibilities and Compliance
Employees are responsible for every action taken under their account. Violations of this Password Policy may lead to disciplinary action and revocation of system access. The IT security team audits password compliance quarterly using anonymous strength sampling and reports the results to the Chief Information Security Officer.
"""

DOCS["vpn-remote-access-policy.md"] = """# VPN and Remote Access Policy

## Purpose
This VPN and Remote Access Policy governs how employees connect to the company network from home, from client sites, and while travelling. The virtual private network encrypts traffic between a remote device and the company data centre so that sensitive data cannot be intercepted on public networks.

## Approved VPN Client and Tunnel Mode
Remote access is permitted only through the company approved VPN client using split tunnelling disabled mode, meaning all traffic from a managed device passes through the encrypted tunnel while connected. Personal VPN products and browser proxy extensions are prohibited on company devices. The VPN gateway enforces certificate based device authentication in addition to user credentials.

## Authentication Requirements for Remote Access
Every VPN session requires multi factor authentication. Users authenticate with their network password plus a one time code from the approved authenticator application. Hardware security keys are required for administrators who access production servers over the VPN. Sessions automatically disconnect after 30 minutes of inactivity and after a maximum continuous session of 12 hours.

## Device Requirements
Only company managed laptops with full disk encryption, current antivirus signatures, and the endpoint detection agent may connect over the VPN. Jailbroken phones, rooted devices, and personal computers without encryption are blocked by the gateway health check. If a device fails the health check, the user is placed in a quarantine network segment that can only reach patching and antivirus update servers.

## Prohibited Activities and Monitoring
Employees must not use the VPN to access the company network from countries under sanction, from public kiosk computers, or while connected to an untrusted open wireless hotspot without first enabling the VPN. All VPN sessions are logged with user identity, source address, connection time, and data volume. The security team reviews VPN logs weekly for impossible travel, where the same account connects from two distant countries within a few hours.

## Support
Remote workers who cannot connect should contact the IT helpdesk. Helpdesk staff will never ask for a VPN password or a one time authentication code, and employees must report any such request as a social engineering attempt.
"""

DOCS["incident-response-playbook.md"] = """# Incident Response Playbook

## Purpose
This Incident Response Playbook describes how the company detects, contains, investigates, and recovers from a security incident. It applies to malware infections, unauthorised access, data loss, denial of service attacks, and insider misuse. The goal is to limit damage, preserve evidence, and restore normal operations safely.

## Severity Classification
Incidents are classified by severity. Severity 1 is a critical incident involving confirmed data breach, ransomware, or outage of a customer facing production system, and requires response within 15 minutes. Severity 2 is a major incident such as a compromised privileged account without confirmed data loss, with response within 1 hour. Severity 3 is a minor incident such as a single malware detection on a workstation, with response within 4 business hours.

## Response Phases
The response follows six phases: preparation, detection and analysis, containment, eradication, recovery, and lessons learned. During containment, analysts isolate affected hosts from the network while keeping them powered on to preserve volatile memory evidence. Eradication removes malware, resets compromised credentials, and closes the entry point. Recovery restores systems from clean backups and monitors them for recurrence.

## Roles and Escalation
The Incident Commander leads the response, coordinates communication, and declares severity. The Security Operations Centre performs triage and forensic analysis. Legal and privacy teams decide whether regulators and affected customers must be notified. For a confirmed personal data breach, notification to the regulator must happen within 72 hours of becoming aware of the breach. Executives are briefed every 4 hours during a Severity 1 incident.

## Evidence Handling
All evidence is handled under chain of custody. Disk images and memory captures are hashed, labelled, and stored in the evidence vault. Analysts work on copies, never on original media. Every action, time, and decision is recorded in the incident ticket so the response can be reconstructed later.

## Post Incident Review
Within 5 business days after closure, the team holds a blameless post incident review. Findings, root cause, and corrective actions are documented, assigned owners, and tracked until completion.
"""

DOCS["phishing-response-playbook.md"] = """# Phishing Response Playbook

## Purpose
This Phishing Response Playbook explains what employees and analysts must do when a suspicious email, message, or website attempts to steal credentials or deliver malware. Phishing is the most common entry point for attackers, so fast reporting materially reduces the chance of a wider compromise.

## How to Report Phishing
Employees who receive a suspicious message must not click links, must not open attachments, and must not reply. They should report the message using the Report Phishing button in the mail client, which forwards the original message with full headers to the security mailbox. If the button is unavailable, employees forward the message as an attachment and then delete it from their inbox.

## Triage by the Security Team
The security operations centre triages a reported phishing message within 30 minutes during business hours. Analysts examine sender domain, authentication results for SPF, DKIM, and DMARC, link destinations, and attachment hashes. If the same campaign reached multiple mailboxes, analysts search and purge all copies from the mail system centrally.

## If an Employee Clicked or Entered Credentials
If an employee clicked a link or entered credentials on a fake page, the account password is reset immediately, all active sessions are revoked, and multi factor authentication tokens are reissued. The workstation is isolated and scanned, browser saved passwords are cleared, and the account is monitored for 14 days for unusual logins or mailbox forwarding rules created by an attacker.

## Indicators and Blocking
Confirmed malicious sender domains, URLs, and file hashes are blocked at the email gateway, web proxy, and endpoint agent within 1 hour of confirmation. The indicators are shared with the threat intelligence feed so peer teams can block the same campaign.

## Awareness Metrics
The security team publishes monthly phishing metrics: number of reports, time to triage, click rate in simulated phishing campaigns, and repeat clicker coaching. Employees who report simulated phish receive positive recognition, reinforcing reporting behaviour rather than punishment.
"""

DOCS["ransomware-response-playbook.md"] = """# Ransomware Response Playbook

## Purpose
This Ransomware Response Playbook provides the specific steps for a ransomware attack in which files are encrypted and a ransom note demands payment, usually in cryptocurrency. Ransomware is always treated as a Severity 1 incident because encryption and data theft often occur together.

## Immediate Containment Steps
On confirming ransomware, immediately disconnect infected machines from the network by removing network cables and disabling wireless, but do not power machines off unless instructed by the forensic team. Disable shared network drives and suspend backup jobs that could overwrite clean backup copies. Block command and control domains at the firewall and preserve the ransom note and a sample encrypted file for analysis.

## Payment Decision
The company policy is not to pay a ransom. Payment does not guarantee decryption, may be illegal where the attacker is under sanction, and funds further attacks. Any exception requires written approval from the Chief Executive, the Legal team, and law enforcement consultation. The decision and its reasoning are recorded in the incident record regardless of outcome.

## Recovery from Backups
Systems are rebuilt from known clean backups stored in the immutable backup repository, which cannot be modified or deleted for 30 days. Before restoration, backups are scanned and a restore point from before the initial infection time is selected. Domain controllers are recovered first, followed by file servers, business applications, and then user workstations, in that priority order.

## Investigation and Notification
Forensic analysts determine the initial access vector, commonly exposed remote desktop, stolen VPN credentials, or a phishing attachment. Because ransomware groups often exfiltrate data before encryption, the team assumes data theft until proven otherwise and Legal assesses notification duties to regulators and customers.

## Communication Rules
Only the designated spokesperson communicates externally about a ransomware incident. Employees must not discuss the attack on social media or with customers who have not been officially notified.
"""

DOCS["data-classification-and-handling-policy.md"] = """# Data Classification and Handling Policy

## Purpose
This Data Classification and Handling Policy defines four data classification levels and how each level must be stored, transmitted, shared, and destroyed. Correct classification helps employees protect sensitive information without slowing down work on public information.

## The Four Classification Levels
Public data is approved for release to anyone, such as marketing material and published annual reports. Internal data is for employees only, such as internal announcements and organisational charts. Confidential data includes customer records, contracts, pricing, and source code, and its disclosure would cause significant harm. Restricted data is the highest level and includes payment card numbers, health records, authentication secrets, and merger plans, whose disclosure would cause severe harm or regulatory penalties.

## Handling Rules for Confidential and Restricted Data
Confidential data must be encrypted in transit and stored only in approved company repositories. Restricted data must also be encrypted at rest, access is limited to named individuals with a documented business need, and it must never be sent to a personal email account or stored on a personal device. Restricted data must not be pasted into public generative AI tools or external file sharing services.

## Labelling Requirements
Documents containing Confidential or Restricted data must carry the classification label in the header and in the file name metadata. Emails containing Restricted data must be marked with the sensitivity label, which automatically applies encryption and prevents forwarding to external recipients.

## Retention and Destruction
When data reaches the end of its retention period, Confidential and Restricted paper records are destroyed by cross cut shredding, and electronic media are wiped using a certified secure erase method or physically destroyed. Certificates of destruction are retained by the records team.

## Data Loss Prevention
The data loss prevention system monitors email and web uploads for credit card numbers, national identity numbers, and Restricted document fingerprints. Blocked transfers generate an alert that the data owner must review within 2 business days.
"""

DOCS["acceptable-use-policy.md"] = """# Acceptable Use Policy

## Purpose
This Acceptable Use Policy sets the rules for using company computers, networks, email, internet access, and software. Its aim is to keep systems secure, keep employees productive, and meet legal obligations, while allowing reasonable personal use that does not interfere with work.

## Permitted and Prohibited Uses
Company systems are provided primarily for business purposes. Limited personal use is permitted outside working hours provided it is lawful, does not consume excessive bandwidth, and does not store personal files in business repositories. Prohibited uses include illegal downloading of copyrighted material, cryptocurrency mining on company hardware, running a private business, accessing offensive material, and attempting to bypass security controls or proxy filters.

## Software Installation Rules
Employees must not install unapproved software, browser extensions, or free file sharing tools on company devices. Software is installed only from the company software centre, which contains vetted and licensed applications. Requests for new software are submitted to IT, which checks licensing, security reviews, and data handling before approval, normally within 5 business days.

## Email and Internet Monitoring
Employees should have no expectation of privacy when using company systems. For security and legal reasons, the company logs internet access, scans email for malware, and may investigate accounts where misuse is suspected. Investigations require approval from Human Resources and the security team and are documented.

## Social Media and External Communication
Only authorised spokespersons may speak for the company on social media or to the press. Employees speaking in a personal capacity must not disclose Confidential information, customer data, or unreleased product plans, and must make clear that views are their own.

## Consequences
Violations of the Acceptable Use Policy may result in removal of access, disciplinary action up to termination, and referral to law enforcement where a crime is suspected. Managers are responsible for ensuring their teams have read and acknowledged this policy annually.
"""

DOCS["access-control-and-privilege-policy.md"] = """# Access Control and Privilege Policy

## Purpose
This Access Control and Privilege Policy implements the principle of least privilege: every user and system should have only the access needed to perform its job, for only as long as it is needed. Limiting privilege reduces the damage from a compromised account and from insider error.

## Account Provisioning and Approval
Access is granted only after a request approved by the employee's manager and by the data or system owner. New accounts are created with a standard role profile for the job function; extra privileges require a separate justification. Shared accounts are prohibited except for approved service accounts owned by a named team.

## Privileged and Administrator Access
Administrator access is granted on a separate privileged account, never on the daily user account. Privileged sessions to production systems go through the privileged access management vault, which records the session and rotates the password after each use. Standing administrator rights on workstations are removed; elevation is granted just in time for a maximum of 8 hours after approval.

## Access Reviews
System owners review user access every quarter for production and financial systems and every 6 months for other systems. Reviewers must confirm each user still needs the access, remove dormant accounts, and evidence the review for auditors. Accounts that are inactive for 45 days are automatically disabled, and accounts inactive for 90 days are deleted after owner confirmation.

## Joiners, Movers, and Leavers
When an employee changes role, old access from the previous role is removed on the transfer date before new access is added. When an employee leaves, all access is revoked on the last working day, and for involuntary departures access is revoked immediately at the time Human Resources notifies IT.

## Authentication Standards
Interactive access to critical systems requires multi factor authentication. Application programming interface keys and service credentials are stored in the secrets vault, rotated at least every 90 days, and never embedded in source code repositories.
"""

DOCS["backup-and-disaster-recovery-policy.md"] = """# Backup and Disaster Recovery Policy

## Purpose
This Backup and Disaster Recovery Policy ensures that company data and systems can be restored after hardware failure, ransomware, accidental deletion, or a site disaster. Backups are only valuable if they are tested, protected, and available when primary systems are not.

## Backup Schedule and the 3-2-1 Rule
The company follows the 3-2-1 backup rule: at least 3 copies of important data, stored on 2 different media types, with 1 copy stored offsite or in a separate cloud region. Production databases receive a full backup daily and transaction log backups every 15 minutes. File servers receive incremental backups daily and a full backup weekly.

## Retention Periods
Daily backups are retained for 30 days, weekly backups for 12 weeks, and monthly backups for 12 months. Financial records backups are retained for 7 years to meet regulatory requirements. Backup media past retention are securely erased and the erasure is logged.

## Immutable and Air Gapped Copies
At least one backup copy is immutable, meaning it cannot be altered or deleted during its retention window, protecting it from ransomware that tries to destroy backups. Critical systems also maintain an air gapped copy that is not continuously connected to the production network.

## Recovery Objectives
The Recovery Time Objective for critical customer systems is 4 hours, meaning service must be restored within 4 hours of a declared disaster. The Recovery Point Objective for those systems is 15 minutes, meaning at most 15 minutes of data may be lost. Less critical internal systems have a Recovery Time Objective of 24 hours and a Recovery Point Objective of 24 hours.

## Testing Requirements
Restore tests are performed monthly for a sample of systems and a full disaster recovery exercise is performed twice per year. A backup job failure must be investigated within 1 business day; two consecutive failures escalate automatically to the infrastructure manager.
"""

DOCS["bring-your-own-device-policy.md"] = """# Bring Your Own Device Policy

## Purpose
This Bring Your Own Device Policy, known as the BYOD Policy, allows employees to use personally owned smartphones and tablets for company email and collaboration tools under controlled conditions. BYOD offers convenience while keeping company data separable and removable from a personal device.

## Enrolment and Management
Personal devices must be enrolled in mobile device management before accessing company email or files. Enrolment creates an encrypted work profile container that separates company applications and data from personal applications. The company can wipe only the work container, not personal photos or messages, when a device is lost or an employee leaves.

## Device Security Requirements
A BYOD device must run a manufacturer supported operating system version, have a screen lock with a PIN of at least 6 digits or biometric unlock, and have device encryption enabled. Jailbroken or rooted devices are blocked automatically. The work profile locks after 5 minutes of inactivity.

## Lost or Stolen Devices
Employees must report a lost or stolen device that contains company data to the IT helpdesk within 4 hours of discovery. The helpdesk will remotely lock the work container and, if recovery appears unlikely, wipe the company data from the device.

## Allowed Applications and Data
Only approved applications inside the work profile may access company data. Copying company text into personal applications is blocked by policy controls, and company documents must not be saved to personal cloud storage such as a personal drive account.

## Costs and Privacy
The company pays a monthly BYOD stipend toward the employee's mobile plan for roles that require mobile access. The company does not monitor personal application usage, location outside work applications, or personal content on an enrolled device.
"""

DOCS["security-awareness-and-training-policy.md"] = """# Security Awareness and Training Policy

## Purpose
This Security Awareness and Training Policy ensures every employee can recognise and respond to modern threats such as phishing, social engineering, tailgating, and unsafe data handling. Technology controls fail when people are not trained, so awareness is treated as a core security control.

## Mandatory Training Schedule
All new employees complete security awareness training within their first 14 days, before receiving access to customer data systems. All staff repeat refresher training annually. Training covers password hygiene, phishing recognition, data classification, incident reporting, and safe remote working. Completion is tracked in the learning management system and reported to managers monthly.

## Role Based Additional Training
Developers complete secure coding training annually, including the OWASP Top Ten vulnerabilities and secure code review practice. Finance staff complete additional training on payment fraud and business email compromise, because attackers target payment processes. System administrators complete privileged access and logging training.

## Phishing Simulations
The security team runs simulated phishing campaigns at least quarterly. Results are used for coaching, not punishment. Employees who click in a simulation complete a short targeted micro training within 7 days. Employees who fail three simulations in a year receive one to one coaching with the security awareness lead.

## Consequences of Non Completion
Employees who do not complete mandatory training within 30 days of the deadline have their access to customer data systems suspended until training is complete. Managers receive escalation notices at 14 days and again at 25 days after a team member's deadline passes.

## Measuring Effectiveness
Effectiveness is measured by training completion rate, phishing report rate, simulation click rate, and the number of real incidents caused by human error each quarter, with targets reviewed by the security governance board.
"""

DOCS["vulnerability-management-policy.md"] = """# Vulnerability Management Policy

## Purpose
This Vulnerability Management Policy defines how the company discovers, prioritises, remediates, and verifies software and configuration vulnerabilities across servers, workstations, network devices, cloud services, and applications.

## Scanning Cadence
Authenticated vulnerability scans run weekly on internal infrastructure and daily on internet facing assets. Web applications are scanned before every major release and penetration tests are performed annually by an independent external firm, and after any major architectural change.

## Remediation Timelines by Severity
Critical vulnerabilities with a CVSS score of 9.0 or higher must be patched within 7 days on internet facing systems and within 14 days internally. High severity vulnerabilities with CVSS from 7.0 to 8.9 must be patched within 30 days. Medium severity vulnerabilities must be patched within 90 days, and Low severity within 180 days. A vulnerability that is known to be actively exploited is treated as Critical regardless of its CVSS score.

## Exceptions and Risk Acceptance
If a patch cannot be applied, the system owner must apply a compensating control such as network isolation or a virtual patch, and submit a risk acceptance signed by the business owner and the Chief Information Security Officer. Risk acceptances expire after 90 days and must be renewed with evidence that remediation is progressing.

## Patching and Change Control
Patches are tested in the staging environment before production deployment for business critical systems. Emergency patches for actively exploited vulnerabilities may use the expedited change process, with retrospective approval within 2 business days.

## Reporting
The vulnerability team publishes a monthly report showing open vulnerabilities by age and severity, systems out of compliance with remediation timelines, and repeat findings. Overdue Critical vulnerabilities are escalated to executive leadership weekly until closed.
"""

DOCS["employee-onboarding-and-offboarding-policy.md"] = """# Employee Onboarding and Offboarding IT Policy

## Purpose
This Employee Onboarding and Offboarding IT Policy, owned jointly by Human Resources and IT, ensures new joiners receive the equipment and access they need on day one, and that departing employees return assets and lose access completely and on time.

## Onboarding Process and Timelines
Human Resources notifies IT of a new joiner at least 5 business days before the start date using the onboarding form. IT prepares a laptop, user account, email mailbox, and standard role based application access so they are ready for the morning of the first day. The hiring manager confirms any non standard software or access in advance; such requests follow the normal access approval process.

## Day One Requirements for New Joiners
On day one, the new employee signs the Acceptable Use Policy, completes identity verification for account activation, enrols in multi factor authentication, and receives the security awareness training link. Buddy and manager introductions, payroll documentation, and benefits enrolment are coordinated by Human Resources during the first week.

## Probation and Access Changes
Access during probation is limited to the standard role profile. Requests for privileged or customer data access for an employee in probation require additional approval from the department head.

## Offboarding Process
When an employee resigns or is terminated, Human Resources notifies IT on the same business day. On the last working day, IT disables the account at the agreed time, revokes VPN and cloud sessions, transfers mailbox and file ownership to the manager, and arranges collection of laptop, access badge, and hardware tokens. Equipment must be returned within 3 business days of the last day; remote employees receive a prepaid courier label.

## Exit Checks
Before the final payroll run, IT confirms in writing that accounts are disabled, company data has been recovered or wiped from personal devices enrolled in BYOD, and no outstanding assets remain assigned to the leaver.
"""

DOCS["data-retention-and-privacy-policy.md"] = """# Data Retention and Privacy Policy

## Purpose
This Data Retention and Privacy Policy explains how long the company keeps personal and business data, why it keeps it, and how individuals can exercise their privacy rights. Keeping data longer than necessary increases breach impact and legal risk, so retention is limited by purpose.

## Retention Schedule
Customer account records are retained for 6 years after the account closes to meet tax and contract law requirements. Support tickets and call recordings are retained for 3 years. Job applicant records for unsuccessful candidates are retained for 12 months and then deleted, unless the candidate consents to a longer talent pool period. Security logs including authentication and VPN logs are retained for 12 months, with the first 90 days in hot searchable storage. Marketing contact data is deleted after 24 months of inactivity.

## Privacy Rights of Individuals
Individuals may request access to their personal data, correction of inaccurate data, deletion where the law allows, and a copy in a portable format. Requests are submitted to the privacy team and must be answered within 30 days. Identity is verified before fulfilling a request to prevent disclosure to an impersonator.

## Legal Hold Exception
When litigation or a regulatory investigation is reasonably anticipated, a legal hold suspends normal deletion for the relevant records. The Legal team issues the hold in writing, and records under hold are preserved until Legal releases the hold, even if their retention period has expired.

## International Transfers and Processors
Personal data is transferred outside its home region only under approved contractual safeguards and only to processors listed in the vendor register. Every new processor that handles personal data undergoes a privacy and security assessment before a contract is signed.

## Breach Notification and Accountability
Suspected personal data breaches are reported immediately to the privacy team, which coordinates with the incident response team. The Data Protection Officer maintains the record of processing activities and reports compliance to the board annually.
"""

for name, text in DOCS.items():
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")
    print(name, len(text.split()))
