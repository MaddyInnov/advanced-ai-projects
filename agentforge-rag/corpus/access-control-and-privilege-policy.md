# Access Control and Privilege Policy

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
