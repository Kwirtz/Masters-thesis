import email
import smtplib


def send():
    msg = email.message_from_string('')
    msg['From'] = "kevin-w@hotmail.fr"
    msg['To'] = "kevin-w@hotmail.fr"
    msg['Subject'] = "Done !"
    
    s = smtplib.SMTP("smtp.live.com",587)
    s.ehlo() # Hostname to send for this command defaults to the fully qualified domain name of the local host.
    s.starttls() #Puts connection to SMTP server in TLS mode
    s.ehlo()
    s.login('kevin-w@hotmail.fr', '0165Noisete')
    
    s.sendmail("kevin-w@hotmail.fr", "kevin-w@hotmail.fr", msg.as_string())
    
    s.quit()