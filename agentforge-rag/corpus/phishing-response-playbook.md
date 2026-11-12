# Phishing Response Playbook

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
