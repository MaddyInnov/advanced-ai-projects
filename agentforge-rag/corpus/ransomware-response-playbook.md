# Ransomware Response Playbook

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
