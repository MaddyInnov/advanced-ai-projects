# Backup and Disaster Recovery Policy

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
