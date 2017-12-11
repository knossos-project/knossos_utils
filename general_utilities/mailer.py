# -*- coding: utf-8 -*-

from general_utilities.dummy_exception import DummyException
import smtplib
from getpass import getpass

from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate   
 import email.encoders as Encoders


class Mailer():
    def __init__(self, smtp_server, use_starttls=False, use_auth=False,
            use_smtps=False, smtp_user='', smtp_pass='', port=25,
            interactive=False):
        """
        Class for sending mail through an SMTP server.

        Parameters
        ----------

        smtp_server : str
            Hostname / IP of SMTP server.

        use_starttls : bool
            Whether to use STARTTLS (if supported by the server).

        use_auth : bool
            Whether to use SMTP AUTH.

        use_smtps : bool
            Whether to use SMTPS, i.e. encrypted SMTP on port 465 (standard).

        smtp_user : str
            Username to use for SMTP authentication.

        smtp_pass : str
            Password to use for SMTP authentication.

        port : int
            Port to connect to. The default (25) may not be appropriate for
            SMTPS.

        interactive : bool
            Whether to interactively prompt for username and password.
        """

        if interactive:
            smtp_user = raw_input('Username: ')
            smtp_pass = getpass('Password: ')

        if use_auth and not smtp_user:
            raise DummyException('Need to specify username for auth.')

        self.smtp_server = smtp_server
        self.use_smtps = use_smtps
        self.use_starttls = use_starttls
        self.use_auth = use_auth
        self.smtp_user = smtp_user
        self.smtp_pass = smtp_pass
        self.port = port

        self.session = None

    def open_session(self):
        if self.use_smtps:
            self.session = smtplib.SMTP_SSL(self.smtp_server, port=self.port)
        else:
            self.session = smtplib.SMTP(self.smtp_server, port=self.port)

        self.session.ehlo()
        if self.use_starttls:
            if self.session.has_extn('starttls'):
                self.session.starttls()
                self.session.ehlo()
            else:
                raise DummyException('TLS not supported by the server.')
        if self.use_auth:
            if self.session.has_extn('auth'):
                self.session.login(self.smtp_user, self.smtp_pass)
            else:
                raise DummyException('SMTP authentication not supported '
                                     'by the server.')

    def close_session(self):
        if self.session:
            self.session.quit()

    def send_mail(self, from_addr, to_addr, subject, message, attachments=[], reply_to=''):
        """
        Send mail after session has been established by open_session.

        Parameters
        ----------

        from_addr : str
            From address.

        to_addr : iterable of str
            Recipient mail addresses.

        subject : str
            Subject line.

        message : str
            Message body.

        attachments : iterable of tuples
            Every tuple in iterable should have the name of the file to attach as
            its first element and the file name to specify in the mail as second
            element.

        reply_to : str
            Reply-To header.
        """
        
        msg = MIMEMultipart()
        msg['From'] = from_addr
        msg['To'] = COMMASPACE.join(to_addr)
        msg['Date'] = formatdate(localtime=True)
        msg['Subject'] = subject
        msg['Reply-To'] = reply_to
        msg.attach(MIMEText(message, _charset='utf-8'))
              
        for attachment, filename in attachments:
            part = MIMEBase('application', "octet-stream")
            part.set_payload(attachment)
            Encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"' % filename)
            msg.attach(part)
        
        self.session.sendmail(from_addr, [to_addr], msg.as_string())
