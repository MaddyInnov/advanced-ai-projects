# Incident Response Playbook

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
