FROM python:3.7
ENV PYTHONUNBUFFERED 1

WORKDIR /code

# Copying the requirements, this is needed because at this point the volume isn't mounted yet
COPY requirements.txt /code/

# Installing requirements, if you don't use this, you should.
# More info: https://pip.pypa.io/en/stable/user_guide/
RUN pip install -r requirements.txt

# Similar to the above, but with just the development-specific requirements
COPY requirements-dev.txt /code/
RUN pip install -r requirements-dev.txt

# Setup SSH with secure root login
RUN apt-get update \
 && apt-get install -y openssh-server netcat \
 && mkdir /var/run/sshd \
 && echo 'root:password' | chpasswd \
 && sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]