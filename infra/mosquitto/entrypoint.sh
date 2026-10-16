#!/bin/sh
# Gera o arquivo de senhas do Mosquitto a partir do ambiente e sobe o broker.
set -eu
: "${MQTT_USERNAME:?MQTT_USERNAME não definido}"
: "${MQTT_PASSWORD:?MQTT_PASSWORD não definido}"
cp /config-src/mosquitto.conf /mosquitto/config/mosquitto.conf
mosquitto_passwd -b -c /mosquitto/config/passwd "$MQTT_USERNAME" "$MQTT_PASSWORD"
chmod 0700 /mosquitto/config/passwd
chown mosquitto:mosquitto /mosquitto/config/passwd 2>/dev/null || true
exec mosquitto -c /mosquitto/config/mosquitto.conf
