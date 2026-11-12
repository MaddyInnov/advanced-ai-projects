# VPN and Remote Access Policy

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
