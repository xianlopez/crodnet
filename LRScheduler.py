from enum import Enum
import os
import logging


class LRPolicies(Enum):
    scheduled = 1
    onCommand = 2


class ScheduledPolicyOpts:
    epochsLRDict = {}


class OnCommandPolicyOpts:
    def __init__(self):
        self.divisionFactor = 2

    def SetOutdir(self, outdir):
        self.outdir = outdir
        self.flagFile = os.path.join(self.outdir, 'KeepLR')


class LRSchedulerOpts:
    def __init__(self, policy):
        self.policy = policy
        self.scheduledPolicyOpts = ScheduledPolicyOpts()
        self.onCommandPolicyOpts = OnCommandPolicyOpts()


class LRScheduler:

    def __init__(self, opts, outdir):
        self.opts = opts
        if self.opts.policy == LRPolicies.onCommand:
            self.opts.onCommandPolicyOpts.SetOutdir(outdir)
            self.CreateLRFlagFile()

    def CreateLRFlagFile(self):
        with open(self.opts.onCommandPolicyOpts.flagFile, 'w') as fid:
            fid.write('Delete this file to divide the learning rate by a factor of ' + str(self.opts.onCommandPolicyOpts.divisionFactor))

    def GetLearningRateAtEpoch_OnCommand(self, previous_lr):
        if os.path.exists(self.opts.onCommandPolicyOpts.flagFile):
            return previous_lr
        else:
            logging.info('LR flag file missing. Reducing LR.')
            self.CreateLRFlagFile()
            return float(previous_lr) / self.opts.onCommandPolicyOpts.divisionFactor

    def GetLearningRateAtEpoch_Scheduled(self, epoch, previous_lr):
        epochsLRDict = self.opts.scheduledPolicyOpts.epochsLRDict
        if len(epochsLRDict) > 0:
            change_epochs = list(epochsLRDict.keys())
            change_epochs.sort()
            if epoch < change_epochs[0]:
                new_lr = previous_lr
            else:
                new_lr = None
                for i in range(len(change_epochs) - 1):
                    if change_epochs[i] <= epoch < change_epochs[i+1]:
                        new_lr = epochsLRDict[change_epochs[i]]
                if new_lr is None:
                    new_lr = epochsLRDict[change_epochs[len(change_epochs) - 1]]
        else:
            new_lr = previous_lr
        return new_lr

    def GetLearningRateAtEpoch(self, epoch, previous_lr):
        if self.opts.policy == LRPolicies.scheduled:
            return self.GetLearningRateAtEpoch_Scheduled(epoch, previous_lr)
        elif self.opts.policy == LRPolicies.onCommand:
            return self.GetLearningRateAtEpoch_OnCommand(previous_lr)
        else:
            raise Exception('Learning rate policy not recognized: ' + str(self.opts.policy))