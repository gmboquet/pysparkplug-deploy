{{/* Common helpers */}}

{{- define "mixle-mlops.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "mixle-mlops.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "mixle-mlops.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/name: {{ include "mixle-mlops.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "mixle-mlops.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mixle-mlops.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "mixle-mlops.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "mixle-mlops.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* True when we still need the local-state PVC (no external DB and/or no cloud object store). */}}
{{- define "mixle-mlops.needsPVC" -}}
{{- if and .Values.persistence.enabled (or (not .Values.database.url) (not .Values.objectStore.url)) -}}true{{- else -}}{{- end -}}
{{- end -}}
